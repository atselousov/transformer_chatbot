#!/bin/bash

IMAGE_NAME='transformer_chatbot'

sudo docker build -t $IMAGE_NAME -f ./environment/Dockerfile .