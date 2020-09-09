FROM gym_robo:latest

ARG DEBIAN_FRONTEND=noninteractive
RUN pip3 install  numpy==1.19 torch==1.5.1 gym==0.15.3 psutil scipy  pandas mpi4py
COPY . /opt/spinningup
RUN cd /opt/spinningup && pip3 install -e .
