import os
import torch

from utils import openai_transformer_config, load_openai_weights
from model import TransformerModel
from trainer import Trainer
from text import BPEVocab
from dataset import FacebookDataset

def main():
    #-----------------------------------------
    parameters_dir = './parameters'
    datasets_dir = './datasets'

    bpe_vocab_path = os.path.join(parameters_dir, 'bpe.vocab')
    bpe_codes_path = os.path.join(parameters_dir, 'bpe.code')
    train_dataset_path = os.path.join(datasets_dir, 'ConvAI2/train_self_revised_no_cands.txt')
    test_dataset_path = os.path.join(datasets_dir, 'ConvAI2/valid_self_revised_no_cands.txt')
    
    batch_size = 8
    lr = 6.25e-5
    lr_warmup = 2000
    n_jobs = 2
    clip_grad = 1
    n_pos_embeddings = 256
    n_segments = 4
    #-----------------------------------------

    vocab = BPEVocab.from_files(bpe_vocab_path, bpe_codes_path)

    config = openai_transformer_config()
    config.n_pos_embeddings = n_pos_embeddings

    transformer = TransformerModel(n_layers=config.n_layers,
                                   n_embeddings=config.n_embeddings,
                                   n_pos_embeddings=config.n_pos_embeddings,
                                   embeddings_size=config.embeddings_size,
                                   padding_idx=vocab.pad_id,
                                   n_heads=config.n_heads,
                                   dropout=config.dropout,
                                   embed_dropout=config.embed_dropout,
                                   attn_dropout=config.attn_dropout,
                                   ff_dropout=config.ff_dropout,
                                   n_segments=n_segments)

    load_openai_weights(transformer.transformer_module, parameters_dir, n_special_tokens=vocab.n_special_tokens)

    train_dataset = FacebookDataset(train_dataset_path, vocab)
    test_dataset = FacebookDataset(test_dataset_path, vocab)

    model_trainer = Trainer(transformer, train_dataset, test_dataset, 
                            batch_size=batch_size, lr=lr, lr_warmup=lr_warmup,
                            n_jobs=n_jobs, clip_grad=clip_grad, device=torch.device('cuda'))

    model_trainer.train(1)

if __name__ == '__main__':
    main()
