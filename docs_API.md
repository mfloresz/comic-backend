# API del backend (contrato para frontends)

Este documento es todo lo que un frontend necesita para hablar con `server.py`.
Nada aquí depende del `index.html` incluido: puedes reimplementar la GUI
en React, móvil, CLI, etc. contra estos endpoints.

Base: `https://<tu-tunel>.trycloudflare.com` (o `http://host:8000` en local/Docker).

## Convenciones

* Imágenes que **envías**: string base64 **crudo o dataURL** (`data:image/...;base64,...`).
  Se aceptan PNG/JPEG/WEBP (PIL lo decodifica).
* Imágenes que **recibes** (`cleaned_image`, `final_image`): base64 **crudo** en el
  formato pedido (`img_fmt`, default `jpeg`). Para mostrar: prefija el dataURL
  según `img_fmt` de la respuesta (`jpeg`→`data:image/jpeg;base64,`,
  `webp`→`data:image/webp;base64,`, `png`→`data:image/png;base64,`).
* JSON siempre UTF-8 (acentos/¿?/CJK sin escapar del lado cliente: `JSON.stringify` basta).
* CORS: `*` en todo (orígenes, métodos, headers).
* Auth: solo si el server arrancó con `CT_TOKEN` → header `Authorization: Bearer <token>`.
* Errores: `{"detail": "..."}` **con CORS** (gracias a `_guard`; un falso
  "CORS Missing" con 500/524 es el error real escondido — lee el body/log).
* Límite práctico: Cloudflare corta a ~100s (524). Mantén `/inpaint` abajo
  con `hd_limit`.

## Esquemas

```ts
type Block = {
  id: number;                 // índice (0..N). El frontend lo usa como key
  xyxy: [x1, y1, x2, y2];     // caja de texto, px sobre la imagen original
  bubble_xyxy: [x1, y1, x2, y2] | null;
  text_class: "text_bubble" | "text_free" | "";
  angle: number; direction: string;   // del font-detector ("vertical"|"")
  text: string;               // OCR (vacío hasta pasar /ocr)
  translation: string;        // traducción (la pone el frontend, ver §5)
  font_color: [r, g, b] | [];
};
```

## Endpoints

### GET /health
`→ {ok, device: "cuda"|"cpu", providers: [...], t}`.
OJO: `providers` lista lo compilado, no lo que cargó. GPU real = inferencias
rápidas + ausencia de `Failed to load ... providers_cuda` en el log.

### GET /fonts
`→ {fonts: ["ComicNeue-Bold", "DejaVuSans", ...]}` (nombres de las `.ttf/.otf`
del servidor). La fuente por defecto (sin pedir ninguna) es la incluida
`fonts/ComicNeue-Bold.ttf`.

### POST /detect — cajas de texto
```json
{ "image": "<b64>", "source_lang": "English", "ocr_lang": "latin" }
→ { "blocks": [Block, ...] }   // text y translation vacíos
```

### POST /ocr — reconoce texto de las cajas dadas
```json
{ "image": "<b64>", "blocks": [Block, ...],
  "source_lang": "English", "ocr_lang": "latin" }
→ { "blocks": [Block, ...] }   // con .text lleno
```
`source_lang`: nombre inglés (`Japanese|Korean|Chinese|English|French|Spanish|Italian|German|Dutch|Russian`).
`soure→iso` lo resuelve el server para ordenar líneas.
`ocr_lang` (modelo PPOCRv5): `ch` (JA/CN) · `en` · `korean` · `latin` (ES/FR/IT/DE/NL) · `eslav` (RU).

### POST /inpaint — borra el texto (devuelve imagen limpia)
```json
{ "image": "<b64>", "blocks": [Block, ...], "hd_limit": 1024, "img_fmt": "webp" }
→ { "cleaned_image": "<b64>", "mask_empty": false, "img_fmt": "webp" }
```
`hd_limit`: lado mayor en px para inferir (default 1024, `0` = original).
Si 524 → baja a 768. Se inpaintean **todos** los bloques recibidos
(la máscara usa sus `xyxy` sin filtrar por texto).
`img_fmt`: `jpeg` (default) | `webp` (~mitad de peso, recomendado tras el túnel) | `png`.

### POST /render — dibuja traducciones sobre la imagen limpia
```json
{ "image": "<b64-limpia>", "blocks": [Block, ...],
  "img_fmt": "webp",
  "font_size": 120, "min_font_size": 10,
  "font_family": "", "font_data": "",
  "color": "#000000", "outline": true, "outline_color": "#FFFFFF",
  "upper_case": false }
→ { "final_image": "<b64>", "img_fmt": "webp" }
```
* `font_size` = **máximo**: busca binariamente el tamaño más grande que quepa
  en cada globo (no existe "tamaño fijo").
* `font_family`: nombre de `/fonts`. `font_data`: `.ttf/.otf` en base64
  (dataURL ok) — tiene prioridad sobre `font_family`.
* Se omiten bloques con `translation` de <2 caracteres.

### POST /process — atajo detect+ocr+inpaint (sin traducir)
```json
{ "image": "<b64>", "blocks": [], "source_lang": "English",
  "ocr_lang": "latin", "hd_limit": 1024, "img_fmt": "webp" }
→ { "blocks": [Block, ...], "cleaned_image": "<b64>", "img_fmt": "webp" }
```
Ignora `blocks` de entrada (detecta de cero).

## Traducción (NO es del servidor)

Va **en el frontend, directo al LLM** (la key nunca toca Colab):

1. Arma `{"block_0": texto, "block_1": texto, ...}` con los `.text`.
2. System prompt (el del server original):
   `You are an expert translator who translates {SRC} to {DST}. ... Return ONLY a JSON object mapping block_0..block_N to translated text. No explanations. If a block is already in {DST} or gibberish, return it as-is.`
3. OpenAI-compatible: `POST {apiBase}/chat/completions`
   `{model, messages:[{system},{user: ctx + "\nTranslate this:\n" + json}], temperature:1}`
   Gemini: `POST .../models/{model}:generateContent?key=...`
   `{system_instruction:{parts:[{text:sys}]}, contents:[{parts:[{text:user}]}]}`.
4. Extrae el primer `{...}` de la respuesta, `JSON.parse`, asigna
   `blocks[i].translation = obj["block_"+i]`.
5. Sigue con `/inpaint` y `/render` (la traducción viaja dentro de `blocks`).

## Flujo mínimo (curl)

```bash
B=https://<tunel>.trycloudflare.com
IMG=$(base64 -w0 pagina.jpg)
curl -s $B/health
BLK=$(curl -s $B/detect -H'Content-Type: application/json' \
  -d "{\"image\":\"$IMG\",\"source_lang\":\"Japanese\"}")
OCR=$(curl -s $B/ocr -H'Content-Type: application/json' \
  -d "{\"image\":\"$IMG\",\"blocks\":$(echo "$BLK"|jq .blocks),\"source_lang\":\"Japanese\",\"ocr_lang\":\"ch\"}")
# ...traduce OCR->JSON con tu LLM, inyecta .translation en cada bloque...
echo "$OCR" | jq '.blocks | {block_0: .[0].text}'
```

## Límites conocidos

* Sin bidi/shaping RTL: traducir **hacia** árabe/hebreo renderiza mal
  (faltan `arabic_reshaper`+`python-bidi`).
* Sin vertical CJK en render (el desktop sí lo tenía vía Qt).
* `/inpaint` a resolución original en imágenes enormes supera el 524:
  usa `hd_limit`.
