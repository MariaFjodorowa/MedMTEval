FROM python:3.8-slim-buster
COPY ./requirements.txt ./
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir unbabel-comet==1.0.1 --use-feature=2020-resolver && \
    pip3 install --no-cache-dir -r  requirements.txt
COPY ./download.py .
RUN python download.py && chmod -R a+r /root/.cache/torch/