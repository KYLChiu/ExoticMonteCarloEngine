FROM ubuntu:20.04

# Non-interactive, otherwise hangs
ENV DEBIAN_FRONTEND=noninteractive

# CMake >= 3.18
RUN apt-get update \
  && apt-get -y install build-essential \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/* \
  && wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /opt/cmake-3.24.1 \
      && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.24.1 \
      && rm /tmp/cmake-install.sh \
      && ln -s /opt/cmake-3.24.1/bin/* /usr/local/bin

# CUDA pre-installation steps
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt-get update

# g++/gcc/cmake/cuda
RUN apt-get install -y g++ gcc cuda

# TODO: find out how to not set these as fixed versions of cuda.
ENV PATH="/usr/local/cuda-12.2/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH}"