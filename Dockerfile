# GPU rentada (RunPod/Vast) con --gpus all. Ajusta el tag CUDA a tu host.
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv fonts-dejavu \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY server.py modules imkit ./

EXPOSE 8000
# CT_TOKEN opcional: Authorization: Bearer <token>
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
