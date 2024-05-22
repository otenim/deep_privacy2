FROM nvcr.io/nvidia/pytorch:22.08-py3

ARG USERNAME=testuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG USERHOME=/home/$USERNAME
ARG WORKDIR=$USERHOME/dev
ENV TORCH_HOME=$USERHOME/.cache
ENV DEBIAN_FRONTEND=noninteractive

RUN groupadd --gid $USER_GID $USERNAME \
&& useradd -m -s /bin/bash --uid $USER_UID --gid $USER_GID $USERNAME \
&& mkdir -p $WORKDIR \
&& chown -R $USER_UID:$USER_GID $USERHOME

WORKDIR $WORKDIR

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6  qt5-default libgtk2.0-dev pkg-config

USER $USERNAME

RUN python -m pip install --user \
    numpy>=1.20 \
    matplotlib \
    cython \
    tensorboard \
    tqdm \
    ninja==1.10.2 \
    ftfy==6.1.3 \
    moviepy \
    pyspng \
    git+https://github.com/hukkelas/DSFD-Pytorch-Inference \
    wandb \ 
    termcolor \
    git+https://github.com/hukkelas/torch_ops.git \
    git+https://github.com/wmuron/motpy@c77f85d27e371c0a298e9a88ca99292d9b9cbe6b \
    git+https://github.com/facebookresearch/detectron2@96c752ce821a3340e27edd51c28a00665dd32a30#subdirectory=projects/DensePose \
    fast_pytorch_kmeans \
    einops_exts  \ 
    einops \ 
    regex \
    setuptools==59.5.0 \
    resize_right==0.0.2 \
    pillow \
    scipy==1.7.1 \
    webdataset==0.2.26 \
    scikit-image \
    timm==0.6.7
RUN python -m pip install --user --no-deps torch_fidelity==0.3.0 clip@git+https://github.com/openai/CLIP.git@b46f5ac7587d2e1862f8b7b1573179d80dcdd620
RUN python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
RUN python -m pip install --user opencv-python==4.5.5.64 opencv-contrib-python==4.5.5.64