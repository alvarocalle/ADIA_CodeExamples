#!/bin/bash

PROJECT_NAME='test-model-r'
PROJECT_VERSION='1.0'
PROJECT_PORT='8000'
DOCKER_OUT_PORT='8000'
DOCKER_IMAGE=$PROJECT_NAME:$PROJECT_VERSION

docker build -t $PROJECT_NAME:$PROJECT_VERSION .

docker rm -f $PROJECT_NAME

docker run -d -p $DOCKER_OUT_PORT:$PROJECT_PORT -p 9000:9000 --name $PROJECT_NAME $PROJECT_NAME:$PROJECT_VERSION

