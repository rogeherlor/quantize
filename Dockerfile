FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    conda clean -afy

COPY requirements.txt /workspace/requirements.txt

WORKDIR /workspace

RUN conda create -y -n quantize python=3.11 && \
    echo "source $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    /bin/bash -c "source $CONDA_DIR/etc/profile.d/conda.sh && conda activate quantize && \
                    conda install -y tensorboard && \
                    pip install -r /workspace/requirements.txt"

CMD ["/bin/bash"]
