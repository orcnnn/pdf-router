import tempfile
import os
import base64
import json
import time
from io import BytesIO

import requests
import datasets
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI, APIConnectionError
from datasets import get_dataset_config_names

from utils import (
    get_prompts,
    get_marker_api_url,
    get_splits,
    ProcessorLabel,
    marker_text_postprocessing,
    vlm_text_postprocessing
)


from vllm import LLM, SamplingParams

load_dotenv()

_CLIENT = None
_MODEL_NAME = None
_PROMPTS = None
_DEBUG = False

def _init_globals(model_name, debug=False, need_http_client=False):
    global _CLIENT, _MODEL_NAME, _PROMPTS, _DEBUG
    _CLIENT = None
    if need_http_client:
        base_url = os.environ.get("VLLM_API_URL")
        api_key  = os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if base_url and api_key:
            _CLIENT = OpenAI(base_url=base_url, api_key=api_key, timeout=None)
        else:
            from loguru import logger
            logger.warning("HTTP VLM client disabled: VLLM_API_URL veya API key bulunamadı.")
    _MODEL_NAME = model_name
    _PROMPTS = get_prompts()
    _DEBUG = debug


def predict_processor(labels):
    if not isinstance(labels, list):
        return ProcessorLabel.NOT_FOUND
    class_labels = {l['class'] for l in labels if isinstance(l, dict) and 'class' in l}

    if "İçindekiler" in class_labels:
        return ProcessorLabel.NOT_FOUND

    if "Tablo" in class_labels or "Soru" in class_labels or "Metin" in class_labels:
        return ProcessorLabel.MARKER
    #if "Ekler" in class_labels or "Resim" in class_labels or "Kapak Sayfası" in class_labels:
    if "Resim/Tablo Açıklaması" in class_labels or "Resim" in class_labels or "Kapak Sayfası" in class_labels:
        return ProcessorLabel.QWEN_VL_25
    return ProcessorLabel.NOT_FOUND

def send_to_marker(sample):
    image = sample['images']
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    temp_filepath = temp_file.name
    try:
        image.save(temp_filepath, format="PNG")
        post_data = {'filepath': temp_filepath}
        marker_api_url = get_marker_api_url()
        logger.debug(f"Sending to marker: {marker_api_url} | path: {temp_filepath}")

        response = requests.post(
            marker_api_url, data=json.dumps(post_data), headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        result = response.json()
        out = result.get('output', '')
        if (out == ""):
            logger.error(f"Marker couldn't dig any text")
        return out
    except requests.exceptions.RequestException as e:
        logger.error(f"Marker API request failed for path {temp_filepath}: {e}")
        return ""
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from Marker: {e}. Response: {response.text}")
        return ""
    finally:
        temp_file.close()
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

def send_to_qwen_vl_25(sample):
    if _CLIENT is None:
        logger.warning("HTTP VLM client init edilmemiş; send_to_qwen_vl_25 atlanıyor.")
        return ""
    logger.debug("Sending to VLM")
    try:
        buffered = BytesIO()
        sample['images'].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        response = _CLIENT.chat.completions.create(
            model=_MODEL_NAME,
            temperature=0,
            messages=[
                {"role": "system", "content": f"{_PROMPTS['system_prompt_1']}"},
                {"role": "user", "content": [
                    {"type": "text", "text": f"{_PROMPTS['user_prompt_0']}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                ]}
            ]
        )
        return response.choices[0].message.content
    except APIConnectionError as e:
        logger.error(f"Could not connect to VLM API: {e}")
        return ""
    except Exception as e:
        logger.error(f"An unexpected error in send_to_qwen_vl_25: {e}")
        return ""

def send_to_marker_map(sample):
    sample['text'] = marker_text_postprocessing(send_to_marker(sample))
    return sample

def send_to_qwen_vl_25_map(sample):
    sample['text'] = vlm_text_postprocessing(send_to_qwen_vl_25(sample))
    return sample

def predict_processor_map(sample):
    try:
        predictions_col = sample.get('predictions')
        labels_list = []
        if not predictions_col:
            sample['processor'] = ProcessorLabel.NOT_FOUND.value
            return sample
        data = json.loads(predictions_col) if isinstance(predictions_col, str) else predictions_col
        if isinstance(data, dict):
            labels_list = data.get('labels', [])
        elif isinstance(data, list):
            labels_list = data
        labels_list = [l for l in labels_list if isinstance(l, dict) and float(l.get('confidence', 0.0)) > 0.9]
        sample['processor'] = predict_processor(labels_list).value
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Could not parse predictions column. Error: {e}. Data: {sample.get('predictions')}. Setting processor to NOT_FOUND.")
        sample['processor'] = ProcessorLabel.NOT_FOUND.value
    return sample

def generate_responses(llm:LLM, prompts, images, temperature=0.2, max_tokens=2048):
    # Ensure that the list of prompts and images are of the same length
    if len(prompts) != len(images):
        raise ValueError(f"The number of prompts must match the number of images. len(images) = {len(images)}, len(prompts) = {len(prompts)}")

    # Placeholder for image input in the prompt
    placeholder = "<|image_pad|>"

    # Prepare the prompts with vision placeholders
    formatted_prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for prompt in prompts
    ]

    # Define the sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_tokens
    )

    # Prepare the inputs for the model
    inputs = [
        {
            'prompt': formatted_prompts[i],
            'multi_modal_data': {'image': images[i]}
        }
        for i in range(len(prompts))
    ]

    # Load the LLM model
    # llm = LLM(model_name, tensor_parallel_size=4)

    # Generate the outputs
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    # Extract the generated text for each output
    generated_texts = []
    for o in outputs:
        generated_text = o.outputs[0].text
        generated_texts.append(generated_text)

    return generated_texts

class PDFRouter:
    def __init__(self, model_name, debug=False, use_vllm=True, use_marker=True):
        self.model_name = model_name
        self.debug = debug
        self.use_marker = use_marker
        self.use_vllm = use_vllm
        if use_vllm:
            self.vlm = LLM(model_name, tensor_parallel_size=2, gpu_memory_utilization=0.8, max_model_len=60000)
        # HTTP client sadece HTTP yolunu kullanacaksanız gerekli.
        need_http_client = not self.use_vllm  # batched vLLM kullanıyorsanız False
        _init_globals(model_name, debug, need_http_client=need_http_client)

        logger.info("PDFRouter initialized.")

    def process_splits(self, ds_name, output_ds_name, start_from_split=None, until_split=None,
                       limit=None, streaming=False, num_proc=2, skip_existing=True, push_mode="overwrite"):
        """
        skip_existing: True ise, çıktı reposunda zaten bulunan split'leri atlar.
        push_mode: "overwrite" -> split'i tamamen değiştirir; "append" -> varsa eski split ile birleştirir (basit dedup).
        """
        all_splits = get_dataset_config_names(ds_name)
        logger.info(f"Found {len(all_splits)} splits in total: {all_splits}")

        # Çıkış reposundaki mevcut split'ler
        existing_out = set(get_splits(output_ds_name) or [])

        def _to_idx(v, default):
            if v is None:
                return default
            if isinstance(v, int):
                return v
            try:
                return all_splits.index(v)
            except ValueError:
                raise ValueError(f"Split '{v}' not found in {all_splits}")

        start_idx = _to_idx(start_from_split, 0)
        end_idx   = _to_idx(until_split, len(all_splits))
        candidates = all_splits[start_idx:end_idx]

        if skip_existing:
            splits_to_process = [s for s in candidates if s not in existing_out]
            skipped = set(candidates) - set(splits_to_process)
            if skipped:
                logger.info(f"Skipping already-pushed splits: {sorted(skipped)}")
        else:
            splits_to_process = candidates

        logger.info(f"Will process splits: {splits_to_process}")

        for i, split_name in enumerate(splits_to_process):
            logger.info(f"--- Processing Split {i+1}/{len(splits_to_process)}: '{split_name}' ---")
            try:
                ds_any = datasets.load_dataset(ds_name, name=split_name, streaming=streaming)
                if isinstance(ds_any, datasets.DatasetDict):
                    key = 'train' if 'train' in ds_any.keys() else list(ds_any.keys())[0]
                    ds = ds_any[key]
                else:
                    ds = ds_any

                if not streaming:
                    if limit is not None:
                        logger.warning(f"LIMIT ENABLED: first {limit} rows of '{split_name}'.")
                        ds = ds.select(range(min(limit, len(ds))))
                    logger.info(f"Loaded split '{split_name}' with {len(ds)} rows.")
                else:
                    logger.info(f"Loaded split '{split_name}' in streaming mode.")
            except Exception as e:
                logger.error(f"Failed to load split '{split_name}'. Skipping. Error: {e}")
                continue

            start_time = time.time()
            logger.info(f"Processing '{split_name}' with num_proc={num_proc}...")

            _map_np = None if streaming else num_proc

            mapped = ds.map(predict_processor_map, num_proc=_map_np)

            _MARKER = ProcessorLabel.MARKER.value
            _VLM = ProcessorLabel.QWEN_VL_25.value

            marker_ds = mapped.filter(lambda x: x['processor'] == _MARKER, num_proc=_map_np)
            vllm_ds   = mapped.filter(lambda x: x['processor'] == _VLM,    num_proc=_map_np)

            processed = []

            if self.use_marker:
                try:
                    if hasattr(marker_ds, '__len__') and len(marker_ds) == 0:
                        logger.info("No samples for Marker.")
                    else:
                        logger.info(f"Sending samples to Marker...")
                        marker_done = marker_ds.map(send_to_marker_map, num_proc=_map_np)
                        processed.append(marker_done)
                except Exception as e:
                    logger.error(f"Marker stage failed for split '{split_name}': {e}")

            if self.use_vllm:
                try:
                    if hasattr(vllm_ds, '__len__') and len(vllm_ds) == 0:
                        logger.info("No samples for VLM.")
                    else:
                        logger.info(f"Sending samples to VLM (batched)...")
                        images  = list(vllm_ds['images'])
                        prompts = [_PROMPTS['user_prompt_1']] * len(images)
                        texts   = generate_responses(self.vlm, prompts, images, temperature=0.0, max_tokens=2048)
                        texts   = [vlm_text_postprocessing(t) for t in texts]
                        vllm_done = vllm_ds.add_column("text", texts)
                        processed.append(vllm_done)
                except Exception as e:
                    logger.error(f"VLM stage failed for split '{split_name}': {e}")

            if not processed:
                logger.warning(f"No samples were processed for split '{split_name}'.")
                continue

            logger.info("Concatenating processed datasets...")
            final_ds = processed[0] if len(processed) == 1 else datasets.concatenate_datasets(processed)

            duration = (time.time() - start_time) / 60
            logger.success(f"Successfully processed split '{split_name}' in {duration:.2f} minutes.")

            # Push: split adı ile DatasetDict push
            try:
                if push_mode == "append":
                    try:
                        old = datasets.load_dataset(output_ds_name, split=split_name)
                        merged = datasets.concatenate_datasets([old, final_ds])

                        # basit dedup örneği (id veya thesis_id varsa)
                        seen = set()
                        def _dedup(row):
                            k = row.get("thesis_id") or row.get("id")
                            if k in seen:
                                return False
                            seen.add(k)
                            return True

                        merged = merged.filter(_dedup)
                        datasets.DatasetDict({split_name: merged}).push_to_hub(repo_id=output_ds_name, private=False)
                    except Exception:
                        datasets.DatasetDict({split_name: final_ds}).push_to_hub(repo_id=output_ds_name, private=False)
                else:
                    datasets.DatasetDict({split_name: final_ds}).push_to_hub(repo_id=output_ds_name, private=False)

                logger.success(f"Successfully pushed '{split_name}'.")
            except TypeError as e:
                logger.error(f"Push failed (arg error). Use DatasetDict push. Error: {e}")
            except Exception as e:
                logger.error(f"Push failed. Error: {e}")
