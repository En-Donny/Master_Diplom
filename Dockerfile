FROM python:3.11-slim

# Настройка окружения Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY ./requirements-torch.txt ./requirements-torch.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements-torch.txt

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY influx ./influx
COPY model_PINN ./model_PINN
COPY utils ./utils
COPY scripts ./scripts

CMD ["python", "-m", "scripts.run_inference_loop"]