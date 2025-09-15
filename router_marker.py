# router_marker.py
import io
from typing import Optional
from PIL import Image
from marker_client import MarkerClient
from utils import get_marker_api_url

_MARKER_CLIENT: Optional[MarkerClient] = None

def init_marker_client(base_url: str,
                       prefer_endpoint: str = "auto",
                       prefer_payload: str = "auto",
                       timeout_connect: int = 2,
                       timeout_read: int = 120):
    global _MARKER_CLIENT
    _MARKER_CLIENT = MarkerClient(
        base_url=base_url,
        timeout_connect=timeout_connect,
        timeout_read=timeout_read,
        prefer_endpoint=prefer_endpoint,
        prefer_payload=prefer_payload,
    )

def send_to_marker(sample) -> str:
    """ sample['images'] -> PIL.Image """
    global _MARKER_CLIENT
    if _MARKER_CLIENT is None:
        # Lazy-init for worker processes (num_proc case)
        init_marker_client(get_marker_api_url())

    img = sample["images"]
    if isinstance(img, Image.Image):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
    else:
        # HF Image feature bazen {'bytes':..., 'path':...} dönebilir
        # burada garantiye alıyoruz:
        if isinstance(img, dict) and "bytes" in img:
            png_bytes = img["bytes"]
        else:
            raise ValueError("sample['images'] must be a PIL.Image or dict with 'bytes'.")

    return _MARKER_CLIENT.call_with_png_bytes(png_bytes) or ""
