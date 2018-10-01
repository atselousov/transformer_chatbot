#!/bin/bash

mkdir checkpoints
wget -O checkpoints/last_checkpoint https://www.dropbox.com/s/cs6zd9yntn6ixea/last_checkpoint?dl=1

wget -O parameters.zip https://www.dropbox.com/s/n2jbjyq32x6jgr6/parameters.zip?dl=1
unzip parameters.zip

./environment/run_reirieval_server.sh
./environment/create_docker_image.sh
