FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/input /app/output /app/test_output

ENV DOCKER=true

CMD ["sh", "-c", "python extract_structure.py && python abc.py && python train_model_lightgbm.py && python main_new.py"]
