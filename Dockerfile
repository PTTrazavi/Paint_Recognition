# Start FROM Nvidia TensorFlow image https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow
FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        git \
        vim \
        curl \
        rsync \
        software-properties-common \
        sudo \
        sqlite3 \
        zip \
        unzip \
        rar \
        unrar \
        apache2-utils \
        nano
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip3 --no-cache-dir install --upgrade pip
COPY requirements.txt .
RUN pip3 --no-cache-dir install -r requirements.txt && \
	rm requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN mkdir /workspace/recognition
WORKDIR /workspace/recognition
ENTRYPOINT bash
