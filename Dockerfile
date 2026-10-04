FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /root/.streamlit && \
    if [ -f config.toml ]; then \
        cp config.toml /root/.streamlit/config.toml; \
    fi

EXPOSE 8501

CMD ["sh", "-c", "streamlit run main.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]