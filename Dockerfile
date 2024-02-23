# FROM ubuntu:latest
#
# RUN apt update && apt upgrade -y
#
# RUN apt install -y -q build-essential python3-pip python3-dev
# RUN pip3 install -U pip setuptools wheel
# RUN pip3 install gunicorn uvloop httptools
#
# COPY requirements.txt /app/requirements.txt
# RUN pip3 install -r /app/requirements.txt
#
# COPY service/ /app
#
# ENV ACCESS_LOG=${ACCESS_LOG:-/proc/1/fd/1}
# ENV ERROR_LOG=${ERROR_LOG:-/proc/1/fd/2}
#
# ENTRYPOINT /usr/local/bin/gunicorn \
#   -b 0.0.0.0:80 \
#   -w 4 \
#   -k uvicorn.workers.UvicornWorker main:app \
#   --chdir /app \
#   --access-logfile "$ACCESS_LOG" \
#   --error-logfile "$ERROR_LOG"

#FROM continuumio/miniconda3
#
#ENV MLFLOW_HOME /opt/mlflow
#ENV MLFLOW_VERSION 1.12.1
#ENV PORT 5000
#
#RUN conda install -c conda-forge mlflow=${MLFLOW_VERSION}
#
#COPY model/ ${MLFLOW_HOME}/model
#
#WORKDIR ${MLFLOW_HOME}
#
#RUN mlflow models prepare-env -m ${MLFLOW_HOME}/model
#
#RUN useradd -d ${MLFLOW_HOME} mlflow
#RUN chown mlflow: ${MLFLOW_HOME}
#USER mlflow
#
FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1
# copy the local requirements.txt file to the
# /app/requirements.txt in the container
# (the /app dir will be created)
COPY ./requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel
# install the packages from the requirements.txt file in the container
RUN pip install -r /app/requirements.txt
# expose the port that uvicorn will run the app
#EXPOSE 8000:8000
# copy the local app/ folder to the /app fodler in the container
COPY ./ /app
# set the working directory in the container to be the /app
WORKDIR /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]