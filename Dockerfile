FROM python:3.8-slim-buster
COPY ./requirements.txt ./
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir unbabel-comet==1.0.1 --use-feature=2020-resolver
RUN pip3 install --no-cache-dir -r  requirements.txt
COPY ru.spm en.spm ./
COPY evaluate.py ./
COPY tests tests
