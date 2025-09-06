# marker_client.py
from __future__ import annotations
import io, os, re, json, tempfile, logging
import requests
from typing import Callable, Optional

log = logging.getLogger("marker-client")

class MarkerClient:
    """
    Tek sefer endpoint & payload keşfi yapar, sonra cache'ler.
    prefer_endpoint: 'auto' | 'exact'  (exact: URL'e ek path ekleme)
    prefer_payload:  'auto' | 'multipart:file' | 'multipart:image' | 'multipart:image_file'
                     | 'json:image_base64' | 'json:filepath'
    """
    def __init__(self,
                 base_url: str,
                 timeout_connect: int = 2,
                 timeout_read: int = 120,
                 prefer_endpoint: str = "auto",
                 prefer_payload: str = "auto"):
        if not base_url:
            raise ValueError("MarkerClient requires a base_url")

        self.raw_base = base_url.rstrip("/")
        self.timeout = (timeout_connect, timeout_read)
        self.prefer_endpoint = prefer_endpoint
        self.prefer_payload = prefer_payload

        self._resolved_url: Optional[str] = None
        self._payload_fn: Optional[Callable[[bytes], Optional[str]]] = None

    # ---------------------- public ----------------------
    def call_with_png_bytes(self, png_bytes: bytes) -> str:
        if self._resolved_url is None or self._payload_fn is None:
            self._auto_discover(png_bytes)
        if self._resolved_url is None or self._payload_fn is None:
            return ""
        try:
            return self._payload_fn(png_bytes) or ""
        except Exception as e:
            log.error("Marker call failed on cached strategy: %s", e)
            return ""

    # ---------------------- discovery ----------------------
    def _auto_discover(self, png_bytes: bytes):
        # 1) Endpoint seçimi
        endpoints = [self.raw_base]
        if self.prefer_endpoint == "auto":
            if not re.search(r"/(marker|predict|extract|process)$", self.raw_base):
                endpoints = [self.raw_base + "/marker", self.raw_base]
        elif self.prefer_endpoint == "exact":
            endpoints = [self.raw_base]
        else:
            endpoints = [self.raw_base]

        # 2) Payload stratejileri
        strategies = self._payload_strategies()
        if self.prefer_payload != "auto":
            # yalnızca seçili yöntemi sıranın başına al
            strategies = [s for s in strategies if s.__name__ == self._map_name(self.prefer_payload)] + \
                         [s for s in strategies if s.__name__ != self._map_name(self.prefer_payload)]

        for url in endpoints:
            for strat in strategies:
                try:
                    out = strat(url, png_bytes)
                except requests.exceptions.RequestException as e:
                    log.debug("Probe network error %s %s", url, e)
                    out = None
                except Exception as e:
                    log.debug("Probe error %s %s", url, e)
                    out = None

                if isinstance(out, str):
                    # başarı
                    self._resolved_url = url
                    self._payload_fn = lambda b, _fn=strat, _url=url: _fn(_url, b)
                    log.info("Marker resolved: url=%s, payload=%s", url, strat.__name__)
                    return

        log.error("Marker auto-discovery failed for base '%s'", self.raw_base)

    # ---------------------- strategies ----------------------
    def _payload_strategies(self):
        def ok_json(r):
            try:
                j = r.json()
            except json.JSONDecodeError:
                return None
            return j.get("output") or j.get("text") or j.get("result") or ""

        def multipart(field_name: str):
            def _fn(url: str, png: bytes):
                files = {field_name: ("page.png", io.BytesIO(png), "image/png")}
                r = requests.post(url, files=files,
                                  headers={"Accept": "application/json"},
                                  timeout=self.timeout)
                if r.ok:
                    return ok_json(r)
                # 404/405 gibi durumlarda None dön, diğerlerini logla
                if r.status_code >= 500:
                    log.debug("Server 5xx on %s (%s)", url, field_name)
                return None
            _fn.__name__ = f"multipart_{field_name}"
            return _fn

    # multipart adayları
        mp_file      = multipart("file")
        mp_image     = multipart("image")
        mp_imagefile = multipart("image_file")

        def json_base64(url: str, png: bytes):
            import base64 as b64
            payload = {"image_base64": b64.b64encode(png).decode()}
            r = requests.post(url, json=payload,
                              headers={"Accept": "application/json"},
                              timeout=self.timeout)
            if r.ok:
                return ok_json(r)
            # 422'de çoğu build "filepath" bekliyor — None dön
            return None
        json_base64.__name__ = "json_image_base64"

        def json_filepath(url: str, png: bytes):
            # bazı Marker sürümleri sadece {"filepath": "..."} kabul ediyor
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                tmp.write(png)
                tmp.flush()
                payload = {"filepath": tmp.name}
                r = requests.post(url, json=payload,
                                  headers={"Accept": "application/json"},
                                  timeout=self.timeout)
                if r.ok:
                    return ok_json(r)
                return None
        json_filepath.__name__ = "json_filepath"

        return [mp_file, mp_image, mp_imagefile, json_base64, json_filepath]

    @staticmethod
    def _map_name(s: str) -> str:
        return s.replace(":", "_")
        