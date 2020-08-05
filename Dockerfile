FROM gym_robo:latest

ARG DEBIAN_FRONTEND=noninteractive
RUN pip3 install  numpy==1.15 torch==1.3.1 gym==0.15.3 pytest psutil scipy mpi4py pandas cloudpickle==1.2.1 seaborn==0.8.1 matplotlib==3.1.1
COPY . /opt/spinningup
RUN cd /opt/spinningup && pip3 install -e .