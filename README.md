# ConvAI2
## Team: Lost in Conversation /2


### Links

BPE vocabulary: https://www.dropbox.com/s/n2jbjyq32x6jgr6/parameters.zip?dl=1

Model checkpoint file: https://www.dropbox.com/s/d6pnsjwfpl238k8/last_checkpoint?dl=1

### Team

* Alexander Tselousov aleksander.tselousov@yandex.ru
* Sergey Golovanov Sergey_XG@mail.ru

### How to run

Unzip BPE vocabulary files into `./parametes` folder and save checkpoint into 
`./checkpoints` folder or use scripts (see below). 

The easiest way to prepare environment is to run script `prepare_environment.sh`.
After that docker container with retrieval server must be run in demon mode and 
image with `transformer_chatbot` must be built.

Retrieval server can be run with script `run_retrieval_servet.sh`. 
Server do not need the internet connection for its work, for connection with 
`transformer_chatbot` port `9200` is used (containers must be in the same docker network).   

After preparations metrics can be evaluated with corresponding `docker_*.sh` scripts or
`*.py` scripts can be used in interactive container run.

Run scripts from the root folder of this repository. 

Usage of docker container for `transformer_chatbot` is not necessary, but 
retrieval server must olways be run for correct work of `transformer_chatbot`.

List of used python modules is in `requirements.txt`. Also `pytorch=0.4.1` is used.

### Metrics

f1: 0.1712

hits@1: 0.174
