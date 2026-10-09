"""comic-backend: FastAPI ONNX-only (deteccion + OCR + inpaint + render).

Vive en Colab / GPU rentada. La GUI es index.html (navegador).
Traduccion LLM se hace en el navegador via fetch (la key nunca pasa por aqui).
"""
import base64
import io
import os
import time
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import traceback as _tb
from functools import wraps


def _guard(fn):
    """Convierte excepciones en HTTP 500 con detalle JSON.

    Sin esto, una excepcion no manejada la responde Starlette FUERA del
    CORSMiddleware (sin cabeceras CORS) y el navegador reporta "CORS",
    ocultando el error real (que esta en /tmp/ct.log de Colab).
    """
    @wraps(fn)
    def inner(*a, **k):
        try:
            return fn(*a, **k)
        except HTTPException:
            raise
        except Exception as e:
            _tb.print_exc()
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return inner

TOKEN = os.environ.get("CT_TOKEN", "")

app = FastAPI(title="comic-translate colab (onnx)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_token(authorization: Optional[str]):
    if not TOKEN:
        return
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="bad token")


def b64_to_rgb(b64: str) -> np.ndarray:
    if "," in b64:  # dataURL
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img)


def rgb_to_b64(arr: np.ndarray, fmt: str = "JPEG", quality: int = 92) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def providers():
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
    except Exception:
        return ["CPUExecutionProvider"], "cpu"
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"
    return ["CPUExecutionProvider"], "cpu"


_PROV, _DEVICE = None, None
_DET = None
_OCR = {}
_INPAINT = None


def get_prov():
    global _PROV, _DEVICE
    if _PROV is None:
        _PROV, _DEVICE = providers()
    return _PROV, _DEVICE


def get_detector():
    global _DET
    if _DET is None:
        from modules.detection.rtdetr_v2_onnx import RTDetrV2ONNXDetection
        prov, dev = get_prov()
        _DET = RTDetrV2ONNXDetection(settings=None)
        _DET.initialize(device=dev, confidence_threshold=0.3)
    return _DET


OCR_LANG_MAP = {
    "japanese": "ch", "chinese": "ch",
    "korean": "ko", "english": "en", "french": "latin",
    "spanish": "latin", "italian": "latin", "german": "latin",
    "dutch": "latin", "russian": "eslav",
    "ch": "ch", "en": "en", "ko": "korean", "latin": "latin",
    "ru": "eslav", "eslav": "eslav",
}


def get_ocr(lang: str = "en"):
    global _OCR
    key = OCR_LANG_MAP.get(str(lang).lower(), "latin")
    if key not in _OCR:
        from modules.ocr.ppocr.engine import PPOCRv5Engine
        prov, dev = get_prov()
        eng = PPOCRv5Engine()
        eng.initialize(lang=key, device=dev)
        _OCR[key] = eng
    return _OCR[key]


def get_inpainter():
    global _INPAINT
    if _INPAINT is None:
        from modules.inpainting.lama import LaMa
        prov, dev = get_prov()
        _INPAINT = LaMa(dev, backend="onnx")
    return _INPAINT


def get_hd_config(limit: int = 1024):
    from modules.inpainting.schema import Config
    # En tunel Cloudflare hay tope ~100s por request: inpaint a resolucion
    # original en imagenes grandes no termina. Resize limita el lado mayor
    # para inferencia y pega solo el area enmascarada a full-res.
    if limit and limit > 0:
        return Config(hd_strategy="Resize", hd_strategy_resize_limit=limit)
    return Config(hd_strategy="Original")


import threading


@app.on_event("startup")
def _preload():
    """Descarga/carga detector e inpainter en background.

    Sin esto, el primer /inpaint paga descarga + carga ONNX dentro del
    request y Cloudflare lo mata con 524 antes de responder.
    """
    def _run():
        try:
            print("[preload] detector...", flush=True)
            get_detector()
            print("[preload] inpainter...", flush=True)
            get_inpainter()
            print("[preload] fuentes...", flush=True)
            _scan_fonts()
            print("[preload] listo", flush=True)
        except Exception as e:
            print("[preload] fallo:", repr(e), flush=True)
    threading.Thread(target=_run, daemon=True).start()


# ---------- blocks <-> dict ----------

def blk_to_dict(blk, i: int) -> dict:
    return {
        "id": i,
        "xyxy": [int(v) for v in np.array(blk.xyxy).tolist()],
        "bubble_xyxy": [int(v) for v in np.array(blk.bubble_xyxy).tolist()] if blk.bubble_xyxy is not None else None,
        "text_class": getattr(blk, "text_class", ""),
        "angle": float(getattr(blk, "angle", 0) or 0),
        "direction": getattr(blk, "direction", "") or "",
        "text": getattr(blk, "text", "") or "",
        "translation": getattr(blk, "translation", "") or "",
        "font_color": list(getattr(blk, "font_color", ()) or ()),
    }


def dict_to_blk(d: dict):
    from modules.utils.textblock import TextBlock
    return TextBlock(
        text_bbox=np.array(d["xyxy"]),
        bubble_bbox=np.array(d["bubble_xyxy"]) if d.get("bubble_xyxy") else None,
        text_class=d.get("text_class", ""),
        angle=d.get("angle", 0),
        text=d.get("text", ""),
        translation=d.get("translation", ""),
        direction=d.get("direction", ""),
        font_color=tuple(d.get("font_color", ()) or ()),
    )


def simple_mask(img: np.ndarray, blks, pad: int = 4) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in blks:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy]
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    try:
        import cv2
        k = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, k, iterations=2)
    except Exception:
        pass
    return mask


# ---------- schemas ----------

class ImgIn(BaseModel):
    image: str = Field(description="base64 o dataURL")
    source_lang: str = "English"
    ocr_lang: str = "en"


class OcrIn(BaseModel):
    image: str
    blocks: list[dict]
    source_lang: str = "English"
    ocr_lang: str = "en"
    hd_limit: int = 1024


class BlocksIn(BaseModel):
    image: str
    blocks: list[dict]
    hd_limit: int = 1024  # lado mayor para inferencia; 0 = original (lento)


class RenderIn(BaseModel):
    image: str  # imagen ya limpia (b64)
    blocks: list[dict]
    font_size: int = 120       # tamaño máximo (crece hasta que llene el globo)
    min_font_size: int = 10
    font_family: str = ""      # nombre de /fonts; "" = DejaVu
    font_data: str = ""        # .ttf/.otf en base64 (subido desde el navegador)
    color: str = "#000000"
    outline: bool = True
    outline_color: str = "#FFFFFF"
    upper_case: bool = False


# ---------- endpoints ----------

@app.get("/health")
def health():
    prov, dev = get_prov()
    return {"ok": True, "device": dev, "providers": prov, "t": time.time()}


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.post("/detect")
@_guard
def detect(body: ImgIn, authorization: Optional[str] = Header(None)):
    _check_token(authorization)
    img = b64_to_rgb(body.image)
    det = get_detector()
    blks = det.detect(img)
    return {"blocks": [blk_to_dict(b, i) for i, b in enumerate(blks)]}


@app.post("/ocr")
@_guard
def ocr(body: OcrIn, authorization: Optional[str] = Header(None)):
    _check_token(authorization)
    from modules.utils.language_utils import language_codes
    img = b64_to_rgb(body.image)
    blks = [dict_to_blk(d) for d in body.blocks]
    lang_en = body.source_lang or "English"
    code = language_codes.get(lang_en, "en")
    for b in blks:
        b.source_lang = code
    eng = get_ocr(body.ocr_lang or code)
    eng.process_image(img, blks)
    return {"blocks": [blk_to_dict(b, i) for i, b in enumerate(blks)]}


@app.post("/inpaint")
@_guard
def inpaint(body: BlocksIn, authorization: Optional[str] = Header(None)):
    _check_token(authorization)
    import time as _t
    import imkit as imk
    t0 = _t.time()
    img = b64_to_rgb(body.image)
    print(f"[inpaint] imagen {img.shape}, bloques={len(body.blocks)}, hd_limit={body.hd_limit}", flush=True)
    blks = [dict_to_blk(d) for d in body.blocks]
    for b in blks:  # la mascara simple usa xyxy; asegura flag de texto
        if not b.text and not b.translation:
            b.text = "x"
    mask = simple_mask(img, blks)
    if int(mask.sum()) == 0:
        return {"cleaned_image": rgb_to_b64(img), "mask_empty": True}
    inp = get_inpainter()
    print(f"[inpaint] modelo listo en { _t.time()-t0:.1f}s, infiriendo...", flush=True)
    out = inp(img, mask, get_hd_config(body.hd_limit))
    out = imk.convert_scale_abs(out)
    print(f"[inpaint] total { _t.time()-t0:.1f}s", flush=True)
    return {"cleaned_image": rgb_to_b64(out), "mask_empty": False}


_FONT_FILES: dict[str, str] = {}
_FONT_CACHE: dict = {}
_NOTO_URL = "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans-Bold.ttf"
REPO_FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "ComicNeue-Bold.ttf")


def _ensure_noto() -> str:
    """Descarga NotoSans-Bold (~200KB, cobertura latina completa incl. ¿é)
    para no depender de que la imagen Colab traiga DejaVu."""
    try:
        from modules.utils.download import models_base_dir
        d = os.path.join(models_base_dir, "fonts")
    except Exception:
        d = os.path.expanduser("~/.fonts")
    p = os.path.join(d, "NotoSans-Bold.ttf")
    if not os.path.exists(p):
        try:
            os.makedirs(d, exist_ok=True)
            import urllib.request
            print("[font] descargando NotoSans-Bold...", flush=True)
            urllib.request.urlretrieve(_NOTO_URL, p)
        except Exception as e:
            print("[font] no se pudo descargar Noto:", repr(e), flush=True)
            return ""
    return p if os.path.exists(p) else ""


def _scan_fonts():
    if _FONT_FILES:
        return _FONT_FILES
    import glob
    for root in ("/usr/share/fonts", "/usr/local/share/fonts",
                 os.path.expanduser("~/.fonts")):
        for ext in ("*.ttf", "*.TTF", "*.otf", "*.OTF", "*.ttc", "*.TTC"):
            for p in glob.glob(os.path.join(root, "**", ext), recursive=True):
                name = os.path.splitext(os.path.basename(p))[0]
                _FONT_FILES.setdefault(name, p)
                _FONT_FILES.setdefault(name.lower(), p)
    return _FONT_FILES


@app.get("/fonts")
def fonts():
    _scan_fonts()
    names = sorted({os.path.splitext(os.path.basename(p))[0]
                    for p in _FONT_FILES.values()})
    return {"fonts": names}


def _font(size: int, family: str = "", font_data: str = ""):
    import hashlib
    size = max(6, int(size))
    raw = b""
    if font_data:
        b64 = font_data.split(",", 1)[1] if "," in font_data else font_data
        try:
            raw = base64.b64decode(b64)
        except Exception:
            raw = b""
    digest = hashlib.sha1(raw).hexdigest()[:12] if raw else ""
    key = (size, family or "", digest)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    f = None
    label = ""
    if raw:
        try:
            f = ImageFont.truetype(io.BytesIO(raw), size)
            label = "ttf-subida"
        except Exception:
            f = None
    if f is None and family:
        p = _scan_fonts().get(family) or _scan_fonts().get(family.lower())
        if p:
            try:
                f = ImageFont.truetype(p, size)
                label = os.path.basename(p)
            except Exception:
                f = None
    if f is None:
        if os.path.exists(REPO_FONT):
            try:
                f = ImageFont.truetype(REPO_FONT, size)
                label = "ComicNeue-Bold(repo)"
            except Exception:
                f = None
    if f is None:
        noto = _ensure_noto()
        if noto:
            try:
                f = ImageFont.truetype(noto, size)
                label = "NotoSans-Bold(descargada)"
            except Exception:
                f = None
    if f is None:
        for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if os.path.exists(p):
                try:
                    f = ImageFont.truetype(p, size)
                    label = os.path.basename(p)
                    break
                except Exception:
                    continue
    if f is None:
        f = ImageFont.load_default()
        label = "PIL-default"
    print(f"[font] {size}px -> {label}", flush=True)
    _FONT_CACHE[key] = f
    return f


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> str:
    if draw.textlength(text, font=font) <= max_w:
        return text
    words = text.split()
    if len(words) > 1:
        lines, cur = [], ""
        for wd in words:
            # parte palabras sueltas que solas exceden el ancho
            while draw.textlength(wd, font=font) > max_w and len(wd) > 1:
                k = len(wd)
                while k > 1 and draw.textlength(wd[:k], font=font) > max_w:
                    k -= 1
                lines.append(wd[:k])
                wd = wd[k:]
            t = (cur + " " + wd).strip()
            if draw.textlength(t, font=font) <= max_w or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = wd
        if cur:
            lines.append(cur)
        return "\n".join(lines)
    # sin espacios (CJK etc): corte por caracteres midiendo
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) <= max_w or not cur:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _fits(draw, text, font, bw, bh):
    wrapped = _wrap(draw, text, font, bw)
    l, t, r, b = draw.multiline_textbbox((0, 0), wrapped, font=font)
    return (r - l) <= bw and (b - t) <= bh, wrapped


@app.post("/render")
@_guard
def render(body: RenderIn, authorization: Optional[str] = Header(None)):
    _check_token(authorization)
    img = b64_to_rgb(body.image)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for d in body.blocks:
        tr = (d.get("translation") or "").strip()
        print(f"[render] bloque: {tr!r}", flush=True)
        if len(tr) < 2:
            continue
        if body.upper_case:
            tr = tr.upper()
        x1, y1, x2, y2 = d["xyxy"]
        bw, bh = max(10, x2 - x1), max(10, y2 - y1)
        # busqueda binaria del tamaño MAS GRANDE que quepa (antes solo
        # encogia desde 40px: en globos grandes el texto quedaba enano).
        lo, hi = body.min_font_size, max(body.font_size, min(400, bh))
        best_size, best_wrapped = lo, tr
        while lo <= hi:
            mid = (lo + hi) // 2
            ok, wrapped = _fits(draw, tr, _font(mid, body.font_family, body.font_data), bw, bh)
            if ok:
                best_size, best_wrapped = mid, wrapped
                lo = mid + 1
            else:
                hi = mid - 1
        font = _font(best_size, body.font_family, body.font_data)
        wrapped = best_wrapped
        l, t, r, b = draw.multiline_textbbox((0, 0), wrapped, font=font)
        tx = x1 + max(0, (bw - (r - l)) // 2)
        ty = y1 + max(0, (bh - (b - t)) // 2)
        if body.outline:
            draw.multiline_text((tx, ty), wrapped, font=font, fill=body.color,
                                align="center", stroke_width=2, stroke_fill=body.outline_color)
        else:
            draw.multiline_text((tx, ty), wrapped, font=font, fill=body.color, align="center")
    return {"final_image": rgb_to_b64(np.array(pil))}


@app.post("/process")
@_guard
def process(body: OcrIn, authorization: Optional[str] = Header(None)):
    """Detect + OCR + inpaint en una sola llamada (sin traducir)."""
    _check_token(authorization)
    import imkit as imk
    img = b64_to_rgb(body.image)
    det = get_detector()
    blks = det.detect(img)
    from modules.utils.language_utils import language_codes
    code = language_codes.get(body.source_lang or "English", "en")
    for b in blks:
        b.source_lang = code
    if blks:
        get_ocr(body.ocr_lang or code).process_image(img, blks)
        for b in blks:
            if not b.text and not b.translation:
                b.text = "x"
        mask = simple_mask(img, blks)
        if int(mask.sum()) > 0:
            out = get_inpainter()(img, mask, get_hd_config(body.hd_limit))
            out = imk.convert_scale_abs(out)
        else:
            out = img
    else:
        out = img
    return {"blocks": [blk_to_dict(b, i) for i, b in enumerate(blks)],
            "cleaned_image": rgb_to_b64(out)}
