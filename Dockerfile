FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install linux-headers-$(uname -r) 
RUN apt-get update && apt-get install -y g++ cmake cuda
