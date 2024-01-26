FROM python:3.10.9-slim-buster

WORKDIR /exp

RUN apt update && apt install -y git gcc g++ make cmake

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN git clone https://github.com/archettialberto/scikit-survival
RUN cd scikit-survival && git submodule update --init
RUN pip install ./scikit-survival

# Enable Jupyter
# RUN pip install jupyter tornado nbconvert matplotlib seaborn

RUN mkdir -p /.local
RUN chmod -R 777 /.local
RUN chmod -R 777 /opt/venv/lib/python3.10/site-packages/pycox/datasets

CMD ["python", "exps.py"]
