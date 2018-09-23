import os
import torch
import random

from utils import openai_transformer_config, load_openai_weights, set_seed, f1_score
from model import TransformerModel
from trainer import Trainer
from text import BPEVocab
from dataset import FacebookDataset


def main():
    #-----------------------------------------
    parameters_dir = './parameters'
    datasets_dir = './datasets'
    checkpoint_dir = './checkpoints'

    last_checkpoint_path = os.path.join(checkpoint_dir, 'last_checkpoint')
    interrupt_checkpoint_path = os.path.join(checkpoint_dir, 'interrupt_checkpoint')

    bpe_vocab_path = os.path.join(parameters_dir, 'bpe.vocab')
    bpe_codes_path = os.path.join(parameters_dir, 'bpe.code')

    convai_train_dataset = ['ConvAI2/train_self_revised_no_cands.txt', 'ConvAI2/train_self_original_no_cands.txt']
    dailydialog_train_dataset = ['DailyDialog/train_dailydialog.txt'] 
    reddit_train_dataset = sorted([os.path.join('Reddit', name) for name in os.listdir(os.path.join(datasets_dir, 'Reddit'))], reverse=True)[:2]

    convai_test_dataset = ['ConvAI2/valid_self_revised_no_cands.txt', 'ConvAI2/valid_self_original_no_cands.txt']
    dailydialog_test_dataset = ['DailyDialog/valid_dailydialog.txt'] 
    reddit_test_dataset = []    


    train_dataset = convai_train_dataset + dailydialog_train_dataset + reddit_train_dataset
    test_dataset = convai_test_dataset + dailydialog_test_dataset + reddit_test_dataset
    train_dataset_paths = [os.path.join(datasets_dir, path) for path in train_dataset]
    test_dataset_paths = [os.path.join(datasets_dir, path) for path in test_dataset]
    

    load_last = True
    n_epochs = 100
    batch_size = 256
    batch_split = 64
    lr = 6.25e-5
    lr_warmup = 3000
    lm_weight = 0.5
    n_jobs = 4
    label_smoothing = 0.1
    clip_grad = None
    n_pos_embeddings = 1024
    n_segments = 3
    max_seq_len = 256
    beam_size = 1
    length_penalty = 0.8
    test_period = 1
    sample = False
    seed = 0
    device = torch.device('cuda')
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
                                   n_segments=n_segments,
                                   sample=sample)

    if not load_last:
        load_openai_weights(transformer.transformer_module, parameters_dir, n_special_tokens=vocab.n_special_tokens)
        print('OpenAI weights loaded')

    train_dataset = FacebookDataset(train_dataset_paths, vocab, n_pos_embeddings - 1)
    test_dataset = FacebookDataset(test_dataset_paths, vocab, n_pos_embeddings - 1)

    model_trainer = Trainer(transformer, train_dataset, test_dataset, batch_size=batch_size,
                            batch_split=batch_split, lr=lr, lr_warmup=lr_warmup, lm_weight=lm_weight,
                            n_jobs=n_jobs, clip_grad=clip_grad, device=device)

    if load_last:
        model_trainer.load_state_dict(torch.load(last_checkpoint_path, map_location=device))
        print('Weights loaded from {}'.format(last_checkpoint_path))
    
    def save_func(epoch):
        torch.save(model_trainer.state_dict(), last_checkpoint_path)  

    def sample_text_func(epoch):
        n_samples = 5
        samples_idxs = random.sample(range(len(test_dataset)), n_samples)
        samples = [test_dataset[idx] for idx in samples_idxs]
        for persona_info, dialog, target in samples:
            contexts = [c.unsqueeze(0).to(model_trainer.device) for c in [persona_info, dialog] if len(c) > 0]
            prediction = model_trainer.model.predict(contexts)[0]
            
            persona_info_str = vocab.ids2string(persona_info[1:-1].tolist())
            dialog_str = vocab.ids2string(dialog.tolist())
            dialog_str = dialog_str.replace(vocab.talker1_bos, '\n\t- ').replace(vocab.talker2_bos, '\n\t -')
            target_str = vocab.ids2string(target[1:-1].tolist())
            prediction_str = vocab.ids2string(prediction)

            print('\n')
            print('Persona info:\n\t{}'.format(persona_info_str))
            print('Dialog:{}'.format(dialog_str))
            print('Target:\n\t{}'.format(target_str))
            print('Prediction:\n\t{}'.format(prediction_str))

    def test_func(epoch):
        if (epoch+1) % test_period == 0:
            metric_funcs = {'f1_score': f1_score}
            model_trainer.test(metric_funcs)

    try:
        model_trainer.train(n_epochs, after_epoch_funcs=[save_func, sample_text_func, test_func])
    except (KeyboardInterrupt, Exception) as e:
        torch.save(model_trainer.state_dict(), interrupt_checkpoint_path) 


if __name__ == '__main__':
    main()
