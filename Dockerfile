FROM python:3.10.9-slim-buster

RUN apt update && apt install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Enable Jupyter
RUN pip install jupyter tornado nbconvert matplotlib seaborn
RUN mkdir -p /.local
RUN chmod -R 777 /.local
RUN chmod -R 777 /opt/venv/lib/python3.10/site-packages/pycox/datasets

WORKDIR /exp

CMD ["python", "exps.py"]
