import os
import torch

from utils import openai_transformer_config, load_openai_weights, set_seed, f1_score, save_model, load_model
from model import TransformerModel
from trainer import Trainer
from text import BPEVocab
from dataset import FacebookDataset

def main():
    #-----------------------------------------
    parameters_dir = './parameters'
    datasets_dir = './datasets'
    checkpoint_dir = './checkpoints'

    bpe_vocab_path = os.path.join(parameters_dir, 'bpe.vocab')
    bpe_codes_path = os.path.join(parameters_dir, 'bpe.code')
    train_dataset_path = os.path.join(datasets_dir, 'ConvAI2/train_self_revised_no_cands.txt')
    test_dataset_path = os.path.join(datasets_dir, 'ConvAI2/valid_self_revised_no_cands.txt')
    
    n_epochs = 100
    batch_size = 128
    batch_split = 16
    lr = 6.25e-5
    lr_warmup = 16000
    n_jobs = 4
    clip_grad = None
    n_pos_embeddings = 1024
    n_segments = 6
    max_seq_len = 320
    beam_size = 1
    length_penalty = 0.8
    test_period = 1
    seed = 0
    #-----------------------------------------
    
    set_seed(seed)

    vocab = BPEVocab.from_files(bpe_vocab_path, bpe_codes_path)

    config = openai_transformer_config()
    config.n_pos_embeddings = n_pos_embeddings
    config.n_embeddings = len(vocab)

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
                                   bos_id=vocab.bos_id,
                                   eos_id=vocab.eos_id,
                                   max_seq_len=max_seq_len,
                                   beam_size=beam_size,  
                                   length_penalty=length_penalty,
                                   n_segments=n_segments)

    load_openai_weights(transformer.transformer_module, parameters_dir, n_special_tokens=vocab.n_special_tokens)

    train_dataset = FacebookDataset(train_dataset_path, vocab)
    test_dataset = FacebookDataset(test_dataset_path, vocab)

    model_trainer = Trainer(transformer, train_dataset, test_dataset, batch_size=batch_size,
                            batch_split=batch_split, lr=lr, lr_warmup=lr_warmup,
                            n_jobs=n_jobs, clip_grad=clip_grad, device=torch.device('cuda'))
    
    def test_func(epoch):
        if epoch % test_period == 0:
            metric_funcs = {'f1_score': f1_score}
            model_trainer.test(metric_funcs)

    def save_func(epoch):
        save_model(model_trainer.model, os.path.join(checkpoint_dir, 'last_checkpoint'))

    model_trainer.train(n_epochs, after_epoch_funcs=[test_func, save_func])

if __name__ == '__main__':
    main()
