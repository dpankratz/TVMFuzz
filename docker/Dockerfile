FROM ubuntu:latest

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git cmake g++ python3 python3-pip wget lsb-release software-properties-common\
    && rm -rf /var/lib/apt/lists/*

#--------------------------------------------------
# Install LLVM binaries
#--------------------------------------------------
#from https://apt.llvm.org/
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 9


#--------------------------------------------------
# Install TVM from source
#--------------------------------------------------
RUN git clone https://github.com/apache/incubator-tvm.git tvm && mkdir tvm/build
WORKDIR "tvm/build"
RUN git submodule init && git submodule update && cp ../cmake/config.cmake . \
	&& sed -i "s|USE_LLVM OFF|USE_LLVM /usr/bin/llvm-config-9|g" config.cmake && cmake ../ && make -j 4

ENV TVM_HOME /
ENV PYTHONPATH $TVM_HOME/python:$TVM_HOME/topi/python:${PYTHONPATH} 

RUN pip3 install --user numpy decorator attrs termcolor && (cd ../python ; python3 setup.py install)
WORKDIR /

#--------------------------------------------------
# Install TVMFuzz
#--------------------------------------------------
RUN git clone https://github.com/dpankratz/TVMFuzz.git tvmfuzz
WORKDIR "tvmfuzz/src"	