FROM nvidia/cuda:11.6.0-devel-ubuntu20.04
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.6"

RUN rm -rf /workspace
WORKDIR /workspace

RUN apt-get update
# RUN apt-get install software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get install -y libgl1-mesa-glx wget bzip2 ca-certificates  build-essential libssl-dev openssl libglib2.0-0 libxext6 libsm6 libxrender1 mercurial subversion libffi-dev && apt-get clean 
RUN apt-get install -y fontconfig libfreetype6 xfonts-75dpi xfonts-base
RUN apt-get install python3-dev
# RUN apt-get install -y python-setuptools
RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
ADD requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
RUN pip install cython
ADD mish-cuda /workspace/mish-cuda
WORKDIR /workspace/mish-cuda
RUN python3 /workspace/mish-cuda/setup.py build install
WORKDIR /workspace
RUN mkdir -p /workspace/input

ADD cropped_egg /workspace/cropped_egg
ADD detect_infer /workspace/detect_infer
ADD final_larva /workspace/final_larva
ADD inference /workspace/inference
ADD models /workspace/models
ADD utils /workspace/utils
ADD run_script /workspace/run_script
ADD detection.py /workspace/detection.py
ADD run.py /workspace/run.py
ADD output.csv /workspace/output.csv
ENTRYPOINT ["python3","/workspace/run.py","--input","/workspace/input"]






