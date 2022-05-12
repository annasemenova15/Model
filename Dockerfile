from ubuntu:20.04
MAINTAINER Anna Semenova
RUN apt-get update -y
COPY . /opt/model_predictor
WORKDIR /opt/model_predictor
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt
CMD python3 model.py

