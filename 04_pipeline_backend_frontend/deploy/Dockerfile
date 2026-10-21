# CPU image for a Hugging Face Space (Docker SDK). Serves the API and the UI on port 7860.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN useradd -m -u 1000 user
WORKDIR /app

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install transformers==5.3.0 peft==0.21.2 fastapi uvicorn python-multipart \
    pypdf python-docx pyyaml numpy pandas scikit-learn

# Bake the backbone into the image so cold starts do not download 2 GB.
ENV HF_HOME=/home/user/.cache/huggingface
RUN python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('xlm-roberta-large'); AutoTokenizer.from_pretrained('xlm-roberta-large')" \
 && chown -R user:user /home/user

# The staged folder mirrors the repository layout (sfaca_paths.py, configs/, NN_*/src, backend, web).
COPY --chown=user:user . /app

USER user
ENV ADAPTER_DIR=/app/adapters PORT=7860 SFACA_MAX_WINDOWS=150 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
EXPOSE 7860
CMD ["python", "04_pipeline_backend_frontend/backend/api.py"]
