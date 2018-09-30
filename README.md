# ConvAI2
## Team: Lost in Conversation /2


### Links

vocabulary: https://www.dropbox.com/s/n2jbjyq32x6jgr6/parameters.zip?dl=1

model: TODO

### Team

* Alexander Tselousov aleksander.tselousov@yandex.ru
* Sergey Golovanov Sergey_XG@mail.ru

### How to run

The easiest way to prepare environment is to run script `prepere_environment.sh`.
After that docker container with retrieval server must be run in demon mode and 
image with `transformer_chatbot` must be built.

Retrieval server can be run with script `run_retrieval_servet.sh`. 
Server do not need the internet connection for its work, connection with 
`transformer_chatbot` goes on through port `9200`.

After preparations metrics can be evaluated with corresponding `.sh` scripts or
in interactive container run.

### Metrics

f1: ?

hits: ?
