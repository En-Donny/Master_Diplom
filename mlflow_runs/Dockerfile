FROM ghcr.io/mlflow/mlflow:v2.7.0

RUN apt-get -y update && \
    apt-get -y install python3-dev default-libmysqlclient-dev build-essential && \
    pip install --upgrade pip && \
    pip install boto3==1.29.2