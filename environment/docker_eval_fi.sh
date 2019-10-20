#!/bin/bash

IMAGE_NAME='transformer_chatbot'

sudo docker run --runtime=nvidia -it --rm -v $(pwd)/parameters:/workspace/parameters   \
                                 -v $(pwd)/checkpoints:/workspace/checkpoints \
