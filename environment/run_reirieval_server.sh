#!/bin/bash

image='atselousov/retrieval_server'
name='retrieval_server'

[[ $(sudo docker ps -f "name=$name" --format '{{.Names}}') == $name ]] || sudo docker run --name $name -p 9200:9200 -d $image