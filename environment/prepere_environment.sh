#!/bin/bash

mkdir checkpoints
wget -O checkpoints/last_checkpoint https://www.dropbox.com/s/cqjzx8jz05iliev/last_checkpoint?dl=1

mkdir parameters
# todo: wget bpe parameters

./environment/run_reirieval_server.sh
./environment/create_docker_image.sh
