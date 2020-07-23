FROM gym_robo:latest

ARG DEBIAN_FRONTEND=noninteractive
RUN pip3 install  numpy==1.15
COPY . /opt/spinningup
RUN cd /opt/spinningup && pip3 install -e .