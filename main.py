import argparse
import yaml
from loguru import logger
from pathlib import Path

from router import PDFRouter  # Aynı klasördeyse bu import yeterli

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}

def main():
    parser = argparse.ArgumentParser(description='PDF Router - Process PDFs with configurable parameters')
    parser.add_argument('config', help='Path to YAML configuration file')
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_name       = cfg.get('model_name', 'Qwen/Qwen2.5-VL-32B-Instruct')
    ds_name          = cfg.get('ds_name')
    output_ds_name   = cfg.get('output_ds_name')
    debug            = bool(cfg.get('debug', False))
    use_vlm          = bool(cfg.get('use_vlm', True))
    use_marker       = bool(cfg.get('use_marker', True))
    n_proc           = int(cfg.get('n_proc', 4))
    streaming        = bool(cfg.get('streaming', False))
    limit            = cfg.get('limit')                  # None or int
    start_from_split = cfg.get('start_from_split')       # name or int
    until_split      = cfg.get('until_split')            # name or int
    skip_existing    = bool(cfg.get('skip_existing', True))
    push_mode        = cfg.get('push_mode', 'overwrite') # or "append"
    vlm_batch_size    = int(cfg.get("vlm_batch_size", 8))
    buffer_size       = int(cfg.get("buffer_size", 256))
    tps               = int(cfg.get("tensor_parallel_size", 2))
    gpu_util          = float(cfg.get("gpu_memory_utilization", 0.7))
    max_model_len     = int(cfg.get("max_model_len", 32000))

    if not ds_name or not output_ds_name:
        raise SystemExit("Config must include 'ds_name' and 'output_ds_name'.")

    router = PDFRouter(
        model_name=model_name,
        debug=debug,
        use_vllm=use_vlm,
        use_marker=use_marker,
        tensor_parallel_size=tps,
        gpu_memory_utilization=gpu_util,
        max_model_len=max_model_len,
        vlm_batch_size=vlm_batch_size,
        buffer_size=buffer_size,
    )

    router.process_splits(
        ds_name=ds_name,
        output_ds_name=output_ds_name,
        start_from_split=start_from_split,
        until_split=until_split,
        limit=limit,
        streaming=streaming,      # 🔴 önemli
        num_proc=cfg.get("n_proc", 4),
        skip_existing=skip_existing,
        push_mode=push_mode,
    )

if __name__ == "__main__":
    main()
