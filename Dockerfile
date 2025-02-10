FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

# Install Python 3.10 and pip
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Set python3.10 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY src/ ./src
COPY ultralytics/ ./ultralytics
COPY datasets/ ./datasets
COPY weights/ ./weights
COPY tempDir/ ./tempDir

# Command to run the application
CMD ["python3", "src/ui.py"]