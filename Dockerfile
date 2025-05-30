FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3 -m pip config set global.break-system-packages true \
	&& pip3 install --no-cache-dir --break-system-packages -r requirements.txt

RUN python3 -m spacy download en_core_web_sm \
	&& python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

COPY ./modelo_sin_contextual.pkl .
COPY modelo.py .
COPY modelo_final.py .

ENV HF_HUB_OFFLINE=1
ENTRYPOINT ["python3", "modelo_final.py", "$inputDataset/dataset.jsonl", "$outputDir"]

