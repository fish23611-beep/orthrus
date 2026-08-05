FROM ubuntu:22.04

# setting up environment variables (timezone for postgresql)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN sed -i \
    -e 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
    -e 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' \
    /etc/apt/sources.list \
    && apt-get -o Acquire::Retries=3 -o Acquire::https::Verify-Peer=false -o Acquire::https::Verify-Host=false update \
    && apt-get -o Acquire::https::Verify-Peer=false -o Acquire::https::Verify-Host=false install -y ca-certificates \
    && update-ca-certificates \
    && apt-get -o Acquire::Retries=3 update \
    && apt-get install -y wget graphviz gnupg lsb-release maven

# installing JDK1.8
RUN apt update && \
    apt install -y openjdk-8-jdk && \
    apt clean

ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
ENV PATH=$JAVA_HOME/bin:$PATH

# installing sudo
RUN apt-get update && apt-get install -y sudo git

# installing a pinned official Miniconda installer (Linux x86_64)
RUN wget --no-verbose https://repo.anaconda.com/miniconda/Miniconda3-py39_24.11.1-0-Linux-x86_64.sh \
    && echo "3ea8373098d72140e08aac9217822b047ec094eb457e7f73945af7c6f68bf6f5  Miniconda3-py39_24.11.1-0-Linux-x86_64.sh" | sha256sum -c - \
    && bash Miniconda3-py39_24.11.1-0-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-py39_24.11.1-0-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:$PATH"

# Make Conda downloads resilient to transient registry disconnects.
RUN conda config --system --set remote_max_retries 10 \
    && conda config --system --set remote_connect_timeout_secs 30 \
    && conda config --system --set remote_read_timeout_secs 180
# installing python libraries
RUN conda create -y -n pids python=3.9 && \
    echo "source /opt/conda/bin/activate pids" >> ~/.bashrc
# https://pythonspeed.com/articles/activate-conda-dockerfile/
SHELL ["conda", "run", "-n", "pids", "/bin/bash", "-c"]
# Activate the environment and install dependencies
RUN conda install -y psycopg2 tqdm && \
    pip install scikit-learn==1.2.0 networkx==2.8.7 xxhash==3.2.0 \
                graphviz==0.20.1 psutil scipy==1.10.1 matplotlib==3.8.4 \
                wandb==0.16.6 chardet==5.2.0 nltk==3.8.1 igraph==0.11.5 \
                cairocffi==1.7.0 wget==3.2

RUN conda install -y "mkl<2024" pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
                pytorch-cuda=11.7 -c pytorch -c nvidia

RUN pip install torch_geometric==2.5.3 --no-cache-dir && \
    pip install pyg_lib==0.2.0 torch_scatter==2.1.1 torch_sparse==0.6.17 \
                torch_cluster==1.6.1 torch_spline_conv==1.2.2 \
                -f https://data.pyg.org/whl/torch-1.13.0+cu117.html --no-cache-dir

RUN pip install gensim==4.3.1 pytz==2024.1 pandas==2.2.2 yacs==0.1.8

RUN pip uninstall -y scipy && pip install scipy==1.10.1 && \
    pip uninstall -y numpy && pip install numpy==1.26.4

# Install test dependencies (pytest, pytest-mock) for container validation
COPY requirements-devcontainer.txt /tmp/requirements-devcontainer.txt
RUN /opt/conda/envs/pids/bin/python -m pip install --no-cache-dir -r /tmp/requirements-devcontainer.txt && \
    rm /tmp/requirements-devcontainer.txt && \
    /opt/conda/envs/pids/bin/python -c "import pytest; import pytest_mock; print('test deps OK')"

WORKDIR /home
COPY . .

RUN [ -f pyproject.toml ] && pip install -e . || echo "No pyproject.toml found, skipping install"
RUN [ -f .pre-commit-config.yaml ] && pre-commit install || echo "No pre-commit found, skipping install"
