#if error,try thiscommand first: docker pull nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

# Install Python 3.10 and pip
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    export PATH=/usr/local/bin:$PATH && \
    ln -s /usr/local/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip && \
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# Set python3.10 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY src/ ./src
COPY ultralytics/ ./ultralytics
COPY datasets/ ./datasets
COPY weights/ ./weights
COPY tempDir/ ./tempDir

# Command to run the application
CMD ["python3", "src/ui.py"]