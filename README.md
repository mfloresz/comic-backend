# comic-backend

Backend ONNX-only (sin PySide, sin GUI) de traducción de cómics + GUI web.
Extraído de `comic-translate`: solo lo que el servidor usa (~1MB de código).

- **Colab / GPU rentada:** `server.py` — FastAPI con `/health /fonts /detect /ocr /inpaint /render /process`.
- **Tu PC (0 dependencias):** `web/index.html` — súbelo con `python3 -m http.server` en `web/`.
- Traducción LLM por `fetch` **local desde el navegador** (la API key nunca pasa por el servidor).

## Modelos (todos ONNX, descarga automática al primer uso)

| Paso | Modelo |
|---|---|
| Detección | RT-DETR-v2 ONNX (+ font-detector ONNX) |
| OCR | PPOCRv5 ONNX (ch/en/korean/latin/eslav) |
| Inpaint | LaMa ONNX (con `hd_limit` para no pasar el tope ~100s del túnel) |
| Render | PIL + NotoSans (auto-descarga) / DejaVu / .ttf subida |

## Colab

1. Sube `comic_translate_colab.ipynb`, runtime **GPU**, ejecutar todo.
2. Copia la URL `https://....trycloudflare.com` a `web/index.html` (campo Servidor).
3. Detect → OCR → Traducir → Inpaint → Render → Descargar.

Detalles que ya nos mordieron (documentados para no repetir):

* **ORT vs CUDA:** el build de `onnxruntime-gpu` debe matchear las libs (`libcublas.so.12`→1.22.0, `.so.13`→latest). El notebook lo elige solo. Build equivocado = caída silenciosa a CPU = 524 en inpaint.
* **Solo [CUDA, CPU]:** `modules/utils/device.py` no ofrece TensorRT/CoreML/OpenVINO; cuando uno no carga, ORT reintenta solo con CPU.
* **Sin PySide:** `modules/utils/language_utils.py` lo importa opcional; `tests/test_no_gui_deps.py` lo garantiza (`python3 -m pytest tests/` sin dependencias).
* **500 sin CORS:** los endpoints devuelven el error en JSON con CORS (`_guard`); un "CORS Missing" con 500/524 casi siempre es el error real escondido — mira el log.

## Docker (renta GPU)

```bash
docker build -t comic-backend .
docker run --gpus all -p 8000:8000 -e CT_TOKEN=secreto comic-backend
```
