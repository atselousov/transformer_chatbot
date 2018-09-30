FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime

RUN  apt-get update \
  && apt-get install -y wget \
  && apt-get install -y vim \
  && apt-get install -y unzip \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get clean && apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

WORKDIR /workspace

COPY ./ ./

RUN pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('vader_lexicon'); nltk.download('perluniprops')"
RUN git clone https://github.com/facebookresearch/ParlAI.git && cd ParlAI && python setup.py develop
RUN python -m spacy download en

EXPOSE 9200

ENTRYPOINT ["/bin/bash"]
