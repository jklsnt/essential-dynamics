FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y git python3.8 python3.8-distutils wget sudo python3.8-dev

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN sudo python3.8 get-pip.py

RUN python3.8 -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3.8 -m pip install fair-esm[esmfold]
RUN python3.8 -m pip install biotite
RUN python3.8 -m pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
RUN python3.8 -m pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

COPY run.py .

ENTRYPOINT ["python3.8", "run.py"]
