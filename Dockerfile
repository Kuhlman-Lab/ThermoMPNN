FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ADD . /home/ThermoMPNN/

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && apt-get install python3 -y\
    && apt-get install python3-pip -y\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r /home/ThermoMPNN/requirements.txt
ENV PYTHONPATH=/home/ThermoMPNN/
ENV THERMO_MPNN=/home/ThermoMPNN/
ENTRYPOINT ["python3", "/home/ThermoMPNN/analysis/custom_inference.py"]
WORKDIR /workdir/
