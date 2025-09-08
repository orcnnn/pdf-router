# utils_marker.py
# -*- coding: utf-8 -*-

import os
import re
import copy
import yaml
import threading
from enum import Enum
from itertools import cycle
from typing import List, Optional

import requests
from dotenv import load_dotenv

# -----------------------------
# Yol & dosya yükleme yardımcıları
# -----------------------------
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ bazı ortamlarda olmayabilir (örn. REPL)
    SCRIPT_DIR = os.getcwd()

PROMPTS_PATH = os.path.join(SCRIPT_DIR, "prompts.yml")

def _load_prompts(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    except FileNotFoundError:
        return {}
    except Exception:
        # YAML hataları vb. durumda boş dön
        return {}

prompts = _load_prompts(PROMPTS_PATH)

# -----------------------------
# Enum / Sınıflandırmalar
# -----------------------------
class ProcessorLabel(Enum):
    NOT_FOUND = 0
    MARKER = 1
    QWEN_VL_25 = 2

CLASSES: List[str] = [
    "Metin",
    "Sayfa Numarası",
    "Resim/Tablo Açıklaması",
    "Resim",
    "Soru",
    "Tablo",
    "İçindekiler",
    "Kapak Sayfası",
    "Header",
    "Footer",
    "Ekler",
    "Diğer",
    "Kaynakça",
    "Kısaltmalar",
]

def get_classes() -> tuple:
    """Dışarıya immutability için tuple verelim."""
    return tuple(CLASSES)

def get_prompts() -> dict:
    """
    Prompts dosyasını her çağrıda diskten yükler.
    Dışarıya derin kopya vererek dışarıda mutasyonun içeriği bozmasını engeller.
    """
    data = _load_prompts(PROMPTS_PATH)
    return copy.deepcopy(data)

# -----------------------------
# .env ve HF API ayarları
# -----------------------------
load_dotenv()

API_TOKEN = os.getenv("HF_TOKEN")
HF_SPLITS_URL = "https://datasets-server.huggingface.co/splits"

def _auth_headers() -> Optional[dict]:
    """Token varsa Authorization başlığını döndürür."""
    return {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else None

def get_splits(ds_name: str) -> Optional[List[str]]:
    """
    HF datasets-server üzerinden split listesini getirir.
    - Başarısızlıkta None döner.
    """
    try:
        r = requests.get(
            HF_SPLITS_URL,
            params={"dataset": ds_name},
            headers=_auth_headers(),
            timeout=15,
        )
        r.raise_for_status()
        splits = (r.json().get("splits") or [])
        names = [s.get("split") for s in splits if isinstance(s, dict) and s.get("split")]
        return names or None
    except requests.RequestException:
        return None

# -----------------------------
# Marker Load Balancer (Round-robin, thread-safe)
# -----------------------------
MARKER_API_URLS = [
    "http://127.0.0.1:8001/marker",
    "http://127.0.0.1:8002/marker",
    "http://127.0.0.1:8003/marker",
    "http://127.0.0.1:8004/marker",
]

class MarkerLoadBalancer:
    """Basit round-robin LB. Çok iş parçacıklı durumlar için kilitli."""
    def __init__(self, urls: List[str]):
        if not urls:
            raise ValueError("MarkerLoadBalancer: en az bir URL gerekli.")
        self._it = cycle(urls)
        self._lock = threading.Lock()
        self._urls = list(urls)

    def get_next_url(self) -> str:
        """Bir sonraki URL'yi döndürür (sadece URL)."""
        with self._lock:
            return next(self._it)

    def get_all_urls(self) -> List[str]:
        """Tüm URL'lerin kopyasını döndürür."""
        return list(self._urls)

# Global LB örneği
_marker_load_balancer = MarkerLoadBalancer(MARKER_API_URLS)

def get_marker_api_url() -> str:
    """Round-robin ile bir sonraki Marker URL'si."""
    return _marker_load_balancer.get_next_url()

def get_all_marker_api_urls() -> List[str]:
    """Tüm Marker URL'lerinin kopyası (fallback vb. için)."""
    return _marker_load_balancer.get_all_urls()

# -----------------------------
# Metin sonrası işleme (post-processing)
# -----------------------------
def marker_text_postprocessing(text: str) -> str:
    """
    Aşağıdakileri temizler:
      1) [num]           -> ""
      2) (num)           -> ""  (1-2 haneli; (2023) gibi tarihleri bozmaz)
      3) ![alt](src)     -> ""
      4) <sup>...</sup>  -> ""
    """
    if not text:
        return text

    # [12] benzeri referanslar
    text = re.sub(r"\[\s*\d+\s*\]", "", text)

    # (1) veya (12). 3+ haneli parantezli sayıları (yıllar vb.) koru
    text = re.sub(r"\((?:\d{1,2})\)", "", text)

    # Görseller: alt metin boş/dolu tüm varyantları kapsa
    # ![...](...) kalıbını tamamen kaldır
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)

    # HTML üstsimge (case-insensitive, multiline)
    text = re.sub(r"<sup\b[^>]*>.*?</sup>", "", text, flags=re.IGNORECASE | re.DOTALL)

    return text

def vlm_text_postprocessing(text: str) -> str:
    """
    ```markdown ...``` veya ``` ...``` bloklarını soyup içeriği bırakır.
    Çoklu blokları da yakalar.
    """
    if not text:
        return text
    return re.sub(r"```(?:markdown)?\s*([\s\S]*?)\s*```", r"\1", text).strip()

# -----------------------------
# Modül dışına ne açıyoruz?
# -----------------------------
__all__ = [
    "ProcessorLabel",
    "get_classes",
    "get_prompts",
    "get_marker_api_url",
    "get_all_marker_api_urls",
    "get_splits",
    "marker_text_postprocessing",
    "vlm_text_postprocessing",
]
