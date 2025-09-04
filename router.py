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
from datasets import get_dataset_config_names, DatasetDict

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
        base_url = os.getenv("VLLM_API_URL")
        api_key = os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if base_url and api_key:
            _CLIENT = OpenAI(base_url=base_url, api_key=api_key, timeout=None)
        else:
            logger.warning("HTTP VLM client disabled: VLLM_API_URL or API key is missing.")
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
    if "Resim/Tablo Açıklaması" in class_labels or "Resim" in class_labels or "Kapak Sayfası" in class_labels:
        return ProcessorLabel.QWEN_VL_25
    return ProcessorLabel.NOT_FOUND


import io, json, os, tempfile, logging
import requests
logger = logging.getLogger(__name__)

def send_to_marker(sample):
    # sample['images'] -> muhtemelen PIL.Image.Image
    image = sample['images']

    # PNG'yi diske yazmadan bellekte hazırla
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    marker_api_url = get_marker_api_url()
    logger.debug("Sending to marker (multipart): %s | bytes=%d",
                 marker_api_url, len(buf.getbuffer()))

    try:
        # Çoğu API alan adını "image" bekler; 422 gelirse "file" ile deneriz
        files = {"image": ("page.png", buf, "image/png")}
        headers = {"Accept": "application/json"}  # opsiyonel
        response = requests.post(marker_api_url, files=files, headers=headers, timeout=(2, 60))

        if response.status_code == 422:
            # Alan adın farklı olabilir; "file" ile bir kez daha dene
            buf.seek(0)
            files = {"file": ("page.png", buf, "image/png")}
            response = requests.post(marker_api_url, files=files, headers=headers, timeout=(2, 60))

        response.raise_for_status()
        result = response.json()

        out = result.get('output', '')
        if out == "":
            logger.error("Marker couldn't dig any text (empty 'output')")
        return out

    except requests.exceptions.RequestException as e:
        logger.error("Marker API request failed (multipart): %s", e)
        return ""
    except json.JSONDecodeError as e:
        logger.error("Failed to decode JSON from Marker: %s. Response: %s", e, getattr(response, "text", ""))
        return ""
def send_to_qwen_vl_25(sample):
    if _CLIENT is None:
        logger.warning("HTTP VLM client not initialized; skipping send_to_qwen_vl_25.")
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
                    {"type": "text", "text": f"{_PROMPTS['user_prompt_1']}"},
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
        # Row dict değilse; string ise JSON parse etmeyi dene, değilse sar
        if not isinstance(sample, dict):
            if isinstance(sample, str):
                try:
                    parsed = json.loads(sample)
                    sample = {"predictions": parsed}
                except Exception:
                    sample = {"predictions": sample}
            else:
                sample = {"predictions": None}

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
        else:
            labels_list = []

        labels_list = [l for l in labels_list if isinstance(l, dict) and float(l.get('confidence', 0.0)) > 0.9]
        sample['processor'] = predict_processor(labels_list).value
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        # sample burada dict olduğuna emin değilsek, korumalı erişelim
        try:
            bad_val = sample.get('predictions') if isinstance(sample, dict) else str(sample)[:200]
        except Exception:
            bad_val = "<unrepr>"
        logger.warning(f"Could not parse predictions column. Error: {e}. Data: {bad_val}. Setting processor to NOT_FOUND.")
        sample = sample if isinstance(sample, dict) else {}
        sample['processor'] = ProcessorLabel.NOT_FOUND.value
    return sample


def generate_responses(llm: LLM, prompts, images, temperature=0.2, max_tokens=2048):
    if len(prompts) != len(images):
        raise ValueError(f"The number of prompts must match the number of images. len(images) = {len(images)}, len(prompts) = {len(prompts)}")

    placeholder = "<|image_pad|>"

    formatted_prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for prompt in prompts
    ]

    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    inputs = [
        {
            'prompt': formatted_prompts[i],
            'multi_modal_data': {'image': images[i]}
        }
        for i in range(len(prompts))
    ]

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    generated_texts = []
    for o in outputs:
        generated_text = o.outputs[0].text
        generated_texts.append(generated_text)

    return generated_texts


class PDFRouter:
    def __init__(self, model_name, debug=False, use_vllm=True, use_marker=True,
                 tensor_parallel_size=2, gpu_memory_utilization=0.7, max_model_len=32000,
                 vlm_batch_size=8, buffer_size=256):
        self.model_name = model_name
        self.debug = debug
        self.use_marker = use_marker
        self.use_vllm = use_vllm
        self.vlm_batch_size = vlm_batch_size
        self.buffer_size = buffer_size

        if use_vllm:
            self.vlm = LLM(
                model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len
            )

        need_http_client = not self.use_vllm
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
        end_idx = _to_idx(until_split, len(all_splits))
        candidates = all_splits[start_idx:end_idx]

        if skip_existing:
            splits_to_process = [s for s in candidates if s not in existing_out]
            skipped = set(candidates) - set(splits_to_process)
            if skipped:
                logger.info(f"Skipping already-pushed splits: {sorted(skipped)}")
        else:
            splits_to_process = candidates

        logger.info(f"Will process splits: {splits_to_process}")

        for split_name in splits_to_process:
            logger.info(f"--- Processing split: {split_name} (streaming={streaming}) ---")

            # ------- KESTİRME: Streaming'de doğrudan split='train' yükle, yoksa otomatik fallback -------
            try:
                if streaming:
                    try:
                        ds = datasets.load_dataset(ds_name, name=split_name, split="train", streaming=True)
                    except Exception:
                        ds_any = datasets.load_dataset(ds_name, name=split_name, streaming=True)
                        if hasattr(ds_any, "keys"):
                            key = 'train' if 'train' in ds_any.keys() else next(iter(ds_any.keys()))
                            ds = ds_any[key]
                        else:
                            ds = ds_any
                else:
                    ds_any = datasets.load_dataset(ds_name, name=split_name, streaming=False)
                    if isinstance(ds_any, DatasetDict):
                        if 'train' in ds_any:
                            ds = ds_any['train']
                        else:
                            ds = ds_any[next(iter(ds_any.keys()))]
                    else:
                        ds = ds_any
            except Exception as e:
                logger.error(f"Failed to load split '{split_name}': {e}")
                continue
            # --------------------------------------------------------------------------------------------

            if not streaming:
                try:
                    if limit is not None:
                        logger.warning(f"LIMIT ENABLED: first {limit} rows of '{split_name}'.")
                        ds = ds.select(range(min(limit, len(ds))))
                    logger.info(f"Loaded split '{split_name}' with {len(ds)} rows.")
                except Exception as e:
                    logger.error(f"Failed to prepare split '{split_name}'. Skipping. Error: {e}")
                    continue

                start_time = time.time()
                logger.info(f"Processing '{split_name}' with num_proc={num_proc}...")

                _map_np = None if streaming else num_proc

                mapped = ds.map(predict_processor_map, num_proc=_map_np)

                _MARKER = ProcessorLabel.MARKER.value
                _VLM = ProcessorLabel.QWEN_VL_25.value

                marker_ds = mapped.filter(lambda x: x['processor'] == _MARKER, num_proc=_map_np)
                vllm_ds = mapped.filter(lambda x: x['processor'] == _VLM, num_proc=_map_np)

                processed = []

                if self.use_marker:
                    try:
                        if hasattr(marker_ds, '__len__') and len(marker_ds) == 0:
                            logger.info("No samples for Marker.")
                        else:
                            logger.info("Sending samples to Marker...")
                            marker_done = marker_ds.map(send_to_marker_map, num_proc=_map_np)
                            processed.append(marker_done)
                    except Exception as e:
                        logger.error(f"Marker stage failed for split '{split_name}': {e}")

                if self.use_vllm:
                    try:
                        if hasattr(vllm_ds, '__len__') and len(vllm_ds) == 0:
                            logger.info("No samples for VLM.")
                        else:
                            logger.info("Sending samples to VLM (batched)...")
                            images = list(vllm_ds['images'])
                            prompts = [_PROMPTS['user_prompt_1']] * len(images)
                            texts = generate_responses(self.vlm, prompts, images, temperature=0.0, max_tokens=2048)
                            texts = [vlm_text_postprocessing(t) for t in texts]
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

                try:
                    if push_mode == "append":
                        try:
                            old = datasets.load_dataset(output_ds_name, split=split_name)
                            merged = datasets.concatenate_datasets([old, final_ds])

                            seen_set = set()

                            def _dedup(row):
                                k = row.get("thesis_id") or row.get("id")
                                if k in seen_set:
                                    return False
                                seen_set.add(k)
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

            else:
                # -----------------------------
                # STREAMING = True : process on the fly
                # -----------------------------
                logger.info("Streaming mode: processing as samples arrive...")
                start_time = time.time()

                out_batch = []          # processed rows waiting to be materialized
                results_chunks = []     # list[Dataset] partial chunks to keep RAM low
                vlm_imgs, vlm_rows = [], []
                seen = 0

                def flush_vlm():
                    nonlocal vlm_imgs, vlm_rows, out_batch
                    if not vlm_imgs:
                        return
                    prompts = [_PROMPTS['user_prompt_1']] * len(vlm_imgs)
                    texts = generate_responses(self.vlm, prompts, vlm_imgs, temperature=0.0, max_tokens=2048)
                    texts = [vlm_text_postprocessing(t) for t in texts]
                    for r, t in zip(vlm_rows, texts):
                        r['text'] = t
                        out_batch.append(r)
                    vlm_imgs, vlm_rows = [], []

                def flush_out_batch():
                    nonlocal out_batch, results_chunks
                    if not out_batch:
                        return
                    results_chunks.append(datasets.Dataset.from_list(out_batch))
                    out_batch = []

                for row in ds:
                    row = predict_processor_map(row)
                    proc = row.get('processor') if isinstance(row, dict) else ProcessorLabel.NOT_FOUND.value

                    has_image = isinstance(row, dict) and ('images' in row) and (row['images'] is not None)

                    if self.use_marker and proc == ProcessorLabel.MARKER.value:
                        if not has_image:
                            logger.warning("Row has no 'images' for MARKER; skipping.")
                        else:
                            row = send_to_marker_map(row)
                            out_batch.append(row)

                    elif self.use_vllm and proc == ProcessorLabel.QWEN_VL_25.value:
                        if not has_image:
                            logger.warning("Row has no 'images' for VLM; skipping.")
                        else:
                            vlm_imgs.append(row['images'])
                            vlm_rows.append(row)
                            if len(vlm_imgs) >= self.vlm_batch_size:
                                flush_vlm()

                    # else: NOT_FOUND veya diğer sınıflar → şimdilik atlıyoruz

                    if len(out_batch) >= self.buffer_size:
                        flush_vlm()
                        flush_out_batch()

                    seen += 1
                    if (limit is not None) and (seen >= limit):
                        break

                # final flush
                flush_vlm()
                flush_out_batch()

                if not results_chunks:
                    logger.warning(f"No processed samples for split '{split_name}'.")
                    continue

                final_ds = results_chunks[0] if len(results_chunks) == 1 else datasets.concatenate_datasets(results_chunks)

                try:
                    if push_mode == "append":
                        try:
                            old = datasets.load_dataset(output_ds_name, split=split_name)
                            merged = datasets.concatenate_datasets([old, final_ds])

                            seen_set = set()

                            def _dedup(row):
                                k = row.get("thesis_id") or row.get("id")
                                if k in seen_set:
                                    return False
                                seen_set.add(k)
                                return True

                            merged = merged.filter(_dedup)
                            datasets.DatasetDict({split_name: merged}).push_to_hub(repo_id=output_ds_name, private=False)
                        except Exception:
                            datasets.DatasetDict({split_name: final_ds}).push_to_hub(repo_id=output_ds_name, private=False)
                    else:
                        datasets.DatasetDict({split_name: final_ds}).push_to_hub(repo_id=output_ds_name, private=False)

                    duration = (time.time() - start_time) / 60
                    logger.success(f"Successfully processed (stream) and pushed split '{split_name}' in {duration:.2f} minutes.")
                except TypeError as e:
                    logger.error(f"Push failed (arg error). Use DatasetDict push. Error: {e}")
                except Exception as e:
                    logger.error(f"Push failed. Error: {e}")

