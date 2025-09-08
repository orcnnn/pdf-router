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
from PIL import Image

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
    logger.info(f"Initializing globals - model_name: {model_name}, debug: {debug}, need_http_client: {need_http_client}")
    
    if need_http_client:
        base_url = os.getenv("VLLM_API_URL")
        api_key = os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        logger.info(f"HTTP client setup - base_url: {base_url}, api_key: {'***' if api_key else 'None'}")
        logger.info(f"All environment variables: VLLM_API_URL={os.getenv('VLLM_API_URL')}, VLLM_API_KEY={os.getenv('VLLM_API_KEY')}")
        
        if base_url and api_key:
            _CLIENT = OpenAI(base_url=base_url, api_key=api_key, timeout=None)
            logger.info("✅ HTTP VLM client initialized successfully")
        else:
            logger.error("HTTP VLM client disabled: VLLM_API_URL or API key is missing.")
            logger.error(f"VLLM_API_URL: {base_url}")
            logger.error(f"API_KEY available: {bool(api_key)}")
    else:
        logger.info("HTTP client not needed - using local vLLM")
        
    _MODEL_NAME = model_name
    _PROMPTS = get_prompts()
    _DEBUG = debug
    logger.info(f"Globals initialized - MODEL_NAME: {_MODEL_NAME}, PROMPTS loaded: {bool(_PROMPTS)}")


def predict_processor(labels):
    logger.debug(f"Predicting processor for labels: {labels}")
    
    if not isinstance(labels, list):
        logger.warning(f"Labels is not a list: {type(labels)}")
        return ProcessorLabel.NOT_FOUND
        
    class_labels = {l['class'] for l in labels if isinstance(l, dict) and 'class' in l}
    logger.debug(f"Extracted class labels: {class_labels}")

    if "İçindekiler" in class_labels:
        logger.info("Content type: İçindekiler -> NOT_FOUND")
        return ProcessorLabel.NOT_FOUND

    if "Tablo" in class_labels or "Soru" in class_labels or "Metin" in class_labels:
        logger.info(f"Content type: {class_labels} -> MARKER")
        return ProcessorLabel.MARKER
        
    if "Resim/Tablo Açıklaması" in class_labels or "Resim" in class_labels or "Kapak Sayfası" in class_labels:
        logger.info(f"Content type: {class_labels} -> QWEN_VL_25")
        return ProcessorLabel.QWEN_VL_25
        
    logger.info(f"Content type: {class_labels} -> NOT_FOUND (no matching classes)")
    return ProcessorLabel.NOT_FOUND


import io, json, base64, logging, requests
logger = logging.getLogger("router")

CANDIDATE_PATHS = ["/marker", "/predict", "/extract", "/process", "/"]
CANDIDATE_FIELDS = ["image", "file", "image_file"]

def send_to_qwen_vl_25(sample):
    logger.info("Starting VLM processing...")
    
    if _CLIENT is None:
        logger.error("HTTP VLM client not initialized; skipping send_to_qwen_vl_25.")
        return ""
        
    logger.info(f"VLM client available, model: {_MODEL_NAME}")
    
    try:
        logger.debug("Converting image to PNG...")
        # Resize image to reduce token count
        img = sample['images']
        if img.size[0] > 512 or img.size[1] > 512:
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
            logger.debug(f"Image resized to {img.size}")
        
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        logger.debug(f"Image converted, size: {len(img_str)} characters")
        
        logger.info("Sending request to VLM API...")
        # Prepare prompts with safe fallbacks (always reload to avoid stale cache)
        prompts_dict = get_prompts() or {}
        system_text = prompts_dict.get('system_prompt_1')
        user_text = prompts_dict.get('user_prompt_1')
        if system_text is None:
            logger.warning("system_prompt_1 missing; using minimal system prompt fallback")
            system_text = "You are a helpful assistant. Output ONLY the Markdown body."
        if user_text is None:
            logger.warning("user_prompt_1 missing; using minimal user prompt fallback")
            user_text = "Metni aynen Markdown’a aktar. Görseli betimle, tekrar etme."

        # Some models (e.g., InternVL) may not support the 'system' role in chat template.
        use_internvl_format = "internvl" in (_MODEL_NAME or "").lower() or ('system_prompt_1' not in prompts_dict)
        if use_internvl_format:
            combined_text = f"{system_text}\n\n{user_text}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": combined_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}},
                    ],
                }
            ]
        else:
            messages = [
                {"role": "system", "content": system_text},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}},
                    ],
                },
            ]

        response = _CLIENT.chat.completions.create(
            model=_MODEL_NAME,
            temperature=0,
            top_p=0.9,
            frequency_penalty=0.3,
            presence_penalty=0.1,
            messages=messages,
        )
        
        result = response.choices[0].message.content
        logger.info(f"✅ VLM processing completed, result length: {len(result) if result else 0}")
        logger.info(f"VLM raw result: {repr(result)}")
        logger.debug(f"VLM result preview: {result[:200] if result else 'Empty'}...")
        return result
        
    except APIConnectionError as e:
        logger.error(f"Could not connect to VLM API: {e}")
        return ""
    except Exception as e:
        logger.error(f"An unexpected error in send_to_qwen_vl_25: {e}")
        logger.exception("Full traceback:")
        return ""


from router_marker import send_to_marker
def send_to_marker_map(sample):
    logger.info("Processing sample with Marker...")
    raw_text = send_to_marker(sample)
    logger.debug(f"Marker raw output length: {len(raw_text) if raw_text else 0}")
    
    processed_text = marker_text_postprocessing(raw_text)
    logger.info(f"Marker processing completed, final text length: {len(processed_text) if processed_text else 0}")
    logger.debug(f"Marker result preview: {processed_text[:200] if processed_text else 'Empty'}...")
    
    sample['text'] = processed_text
    sample['processor_used'] = 'marker'
    return sample


def send_to_qwen_vl_25_map(sample):
    logger.info("Processing sample with VLM...")
    raw_text = send_to_qwen_vl_25(sample)
    logger.info(f"VLM raw output length: {len(raw_text) if raw_text else 0}")
    logger.info(f"VLM raw output: {repr(raw_text)}")
    
    processed_text = vlm_text_postprocessing(raw_text)
    logger.info(f"VLM processing completed, final text length: {len(processed_text) if processed_text else 0}")
    logger.info(f"VLM final output: {repr(processed_text)}")
    logger.debug(f"VLM result preview: {processed_text[:200] if processed_text else 'Empty'}...")
    
    sample['text'] = processed_text
    sample['processor_used'] = 'vlm'
    return sample


def predict_processor_map(sample):
    logger.debug(f"Processing sample for classification: {type(sample)}")
    
    try:
        # Row dict değilse; string ise JSON parse etmeyi dene, değilse sar
        if not isinstance(sample, dict):
            logger.debug(f"Sample is not dict, converting...")
            if isinstance(sample, str):
                try:
                    parsed = json.loads(sample)
                    sample = {"predictions": parsed}
                    logger.debug("Successfully parsed string sample as JSON")
                except Exception as e:
                    logger.debug(f"Failed to parse string as JSON: {e}")
                    sample = {"predictions": sample}
            else:
                logger.debug(f"Sample type {type(sample)} not supported, setting predictions to None")
                sample = {"predictions": None}

        predictions_col = sample.get('predictions')
        logger.debug(f"Predictions column type: {type(predictions_col)}")
        
        labels_list = []
        if not predictions_col:
            logger.warning("No predictions column found, setting processor to NOT_FOUND")
            sample['processor'] = ProcessorLabel.NOT_FOUND.value
            return sample

        data = json.loads(predictions_col) if isinstance(predictions_col, str) else predictions_col
        logger.debug(f"Parsed data type: {type(data)}")
        
        if isinstance(data, dict):
            labels_list = data.get('labels', [])
            logger.debug(f"Extracted labels from dict: {len(labels_list)} items")
        elif isinstance(data, list):
            labels_list = data
            logger.debug(f"Using data as labels list: {len(labels_list)} items")
        else:
            logger.warning(f"Unexpected data type: {type(data)}")
            labels_list = []

        # Filter by confidence
        original_count = len(labels_list)
        labels_list = [l for l in labels_list if isinstance(l, dict) and float(l.get('confidence', 0.0)) > 0.85]
        logger.debug(f"Filtered labels by confidence > 0.5: {original_count} -> {len(labels_list)}")
        
        processor = predict_processor(labels_list)
        sample['processor'] = processor.value
        
        # Set processor_used based on classification
        if processor == ProcessorLabel.MARKER:
            sample['processor_used'] = 'marker'
        elif processor == ProcessorLabel.QWEN_VL_25:
            sample['processor_used'] = 'vlm'
        else:
            sample['processor_used'] = 'none'
            
        logger.info(f"Sample classified as: {processor.name} ({processor.value}) -> processor_used: {sample['processor_used']}")
        
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        # sample burada dict olduğuna emin değilsek, korumalı erişelim
        try:
            bad_val = sample.get('predictions') if isinstance(sample, dict) else str(sample)[:200]
        except Exception:
            bad_val = "<unrepr>"
        logger.error(f"Could not parse predictions column. Error: {e}. Data: {bad_val}. Setting processor to NOT_FOUND.")
        logger.exception("Full traceback:")
        sample = sample if isinstance(sample, dict) else {}
        sample['processor'] = ProcessorLabel.NOT_FOUND.value
        sample['processor_used'] = 'none'
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

from utils import get_marker_api_url
from router_marker import init_marker_client
class PDFRouter:
    def __init__(self, model_name, debug=False, use_vllm=True, use_marker=True,
                 tensor_parallel_size=2, gpu_memory_utilization=0.7, max_model_len=32000,
                 vlm_batch_size=8, buffer_size=256):
        logger.info("=" * 60)
        logger.info("INITIALIZING PDF ROUTER")
        logger.info("=" * 60)
        logger.info(f"Model name: {model_name}")
        logger.info(f"Debug mode: {debug}")
        logger.info(f"Use VLM: {use_vllm}")
        logger.info(f"Use Marker: {use_marker}")
        logger.info(f"Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"GPU memory utilization: {gpu_memory_utilization}")
        logger.info(f"Max model length: {max_model_len}")
        logger.info(f"VLM batch size: {vlm_batch_size}")
        logger.info(f"Buffer size: {buffer_size}")
        
        self.model_name = model_name
        self.debug = debug
        self.use_marker = use_marker
        self.use_vllm = use_vllm
        self.vlm_batch_size = vlm_batch_size
        self.buffer_size = buffer_size

        if self.use_marker:
            logger.info("Initializing Marker client...")
            marker_url = get_marker_api_url()
            logger.info(f"Marker API URL: {marker_url}")
            # MARKER_API_URL tam endpoint ise (örn "...:8003/marker") prefer_endpoint='exact' seçebilirsin.
            init_marker_client(
                base_url=marker_url,      # env'den geliyor
                prefer_endpoint="auto",             # 'exact' yaparsan ek path denenmez
                prefer_payload="auto"               # istersen 'json:filepath' zorunlu kıl
            )
            logger.info("✅ Marker client initialized")

        if use_vllm:
            logger.info("Initializing local vLLM...")
            self.vlm = LLM(
                model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len
            )
            logger.info("✅ Local vLLM initialized")

        need_http_client = not self.use_vllm
        logger.info(f"HTTP client needed: {need_http_client}")
        _init_globals(model_name, debug, need_http_client=need_http_client)
        logger.info("✅ PDFRouter initialization completed!")
        logger.info("=" * 60)

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
                
                elif not self.use_vllm and hasattr(vllm_ds, '__len__') and len(vllm_ds) > 0:
                    # HTTP client processing for batch mode
                    try:
                        logger.info("Sending samples to VLM (HTTP client)...")
                        vllm_done = vllm_ds.map(send_to_qwen_vl_25_map, num_proc=_map_np)
                        processed.append(vllm_done)
                    except Exception as e:
                        logger.error(f"VLM HTTP stage failed for split '{split_name}': {e}")

                if not processed:
                    logger.warning(f"No samples were processed for split '{split_name}'.")
                    continue

                logger.info("Concatenating processed datasets...")
                final_ds = processed[0] if len(processed) == 1 else datasets.concatenate_datasets(processed)

                duration = (time.time() - start_time) / 60
                logger.info(f"✅ Successfully processed split '{split_name}' in {duration:.2f} minutes.")

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

                    logger.info(f"✅ Successfully pushed '{split_name}'.")
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
                    
                    logger.info(f"Flushing VLM batch with {len(vlm_imgs)} images...")
                    
                    if self.use_vllm:
                        # Local vLLM processing
                        prompts = [_PROMPTS['user_prompt_1']] * len(vlm_imgs)
                        texts = generate_responses(self.vlm, prompts, vlm_imgs, temperature=0.0, max_tokens=2048)
                        texts = [vlm_text_postprocessing(t) for t in texts]
                    else:
                        # HTTP client processing
                        texts = []
                        for i, img in enumerate(vlm_imgs):
                            logger.debug(f"Processing VLM image {i+1}/{len(vlm_imgs)}")
                            sample = {'images': img}
                            text = send_to_qwen_vl_25(sample)
                            processed_text = vlm_text_postprocessing(text)
                            texts.append(processed_text)
                    
                    for r, t in zip(vlm_rows, texts):
                        r['text'] = t
                        out_batch.append(r)
                    
                    logger.info(f"VLM batch flushed, processed {len(texts)} samples")
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
                    
                    elif not self.use_vllm and proc == ProcessorLabel.QWEN_VL_25.value:
                        # HTTP client processing - process immediately
                        if not has_image:
                            logger.warning("Row has no 'images' for VLM; skipping.")
                        else:
                            logger.info("Processing VLM sample with HTTP client...")
                            row = send_to_qwen_vl_25_map(row)
                            out_batch.append(row)

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
                    logger.info(f"✅ Successfully processed (stream) and pushed split '{split_name}' in {duration:.2f} minutes.")
                except TypeError as e:
                    logger.error(f"Push failed (arg error). Use DatasetDict push. Error: {e}")
                except Exception as e:
                    logger.error(f"Push failed. Error: {e}")

