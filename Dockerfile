FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 
# 12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda.sh && \
    bash /root/miniconda.sh -b -p $CONDA_DIR && \
    rm /root/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy && \
    $CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    $CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Copy requirements
COPY requirements.txt /workspace/requirements.txt

# Set working directory
WORKDIR /workspace

# Create environment and install Python/tensorboard/pip packages
RUN $CONDA_DIR/bin/conda create -y -n quantize python=3.11 && \
    echo "source $CONDA_DIR/etc/profile.d/conda.sh" >> /root/.bashrc && \
    /bin/bash -c "source $CONDA_DIR/etc/profile.d/conda.sh && \
                  conda activate quantize && \
                  conda install -y tensorboard && \
                  pip install -r /workspace/requirements.txt"

CMD ["/bin/bash"]
