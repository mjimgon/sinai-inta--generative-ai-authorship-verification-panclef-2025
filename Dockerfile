FROM python:3.12

RUN pip3 install click tira

ADD modelo_final.py /
ADD modelo_tfid_todo.pkl /
ADD val.jsonl / 
ADD requirements.txt /
ADD modelo.py /

RUN pip3 install -r requirements.txt
RUN python -m spacy download en_core_web_sm
RUN pip install xgboost


RUN pip3 install --upgrade tira
RUN tira-cli login --token 1410a8fb6ebef8ec2bf7b15178ff269b539cf158e4ef54019113fe5316713161
RUN tira-cli verify-installation


#ENTRYPOINT [ "/modelo_final.py" ]