FROM ubuntu:22.04

RUN apt-get update
RUN apt-get upgrade
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y ; apt update & DEBIAN_FRONTEND=noninteractive  apt install -y python3.12  git
RUN apt-get install -y python3-pip
RUN apt-get install -y python3-venv

# Crea el entorno virtual en .venv
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
RUN python3 -m venv .venv

ADD requirements.txt .
# Change the default shell to bash
SHELL ["/bin/bash", "-c"]
# Activa el entorno y actualiza pip + instala requirements
RUN source .venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
RUN source .venv/bin/activate && python -m spacy download en_core_web_sm


COPY nltk_data /nltk_data
ENV NLTK_DATA=/nltk_data

COPY .git .
COPY modelo_tfid_todo.pkl .
COPY val.jsonl .
COPY modelo.py .
COPY modelo_final.py .




