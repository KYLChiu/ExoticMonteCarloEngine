FROM ubuntu:20.04
# Install wget
RUN apt-get update
RUN apt-get install -y wget

# CUDA pre-installation steps
RUN apt-get install -y software-properties-common
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt-get update

# g++/gcc/cmake/cuda

RUN apt-get install -y g++ gcc cmake cuda