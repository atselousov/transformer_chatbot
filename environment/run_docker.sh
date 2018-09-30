#!/bin/bash

sudo nvidia-docker run -it --rm -v /home/alex_sr/Work/TEMP/convai2/parameters:/workspace/parameters   \
                                -v /home/alex_sr/Work/TEMP/convai2/checkpoints:/workspace/checkpoints \
                                -v /home/alex_sr/Work/TEMP/convai2/datasets:/workspace/datasets       \
                                --network host sub
