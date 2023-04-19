FROM python:3.10.9-slim-buster

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN chmod -R 777 /usr/local/lib/python3.10/site-packages/pycox/datasets

WORKDIR /exp

CMD ["bash"]
