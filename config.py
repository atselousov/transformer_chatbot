from attrdict import AttrDict
from model.utils import openai_transformer_config


def get_model_config():
    default_config = openai_transformer_config()
    config = AttrDict({'bpe_vocab_path': './parameters/bpe.vocab',
                       'bpe_codes_path': './parameters/bpe.code',
                       'checkpoint_path': './checkpoints/last_checkpoint', 
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 128,
                       'beam_size': 1,
                       'diversity_coef': 0,
                       'diversity_groups': 1, 
                       'annealing_topk': None,
                       'annealing': 0,
                       'length_penalty': 0.6,
                       'n_segments': None})

    if config.annealing_topk is not None:
        assert config.annealing_topk > config.beam_size
    
    assert config.diversity_coef >= 0
    assert config.beam_size % config.diversity_groups == 0

    return config


def get_trainer_config():
    config = AttrDict({'n_epochs': 100,
                       'batch_size': 256,
                       'batch_split': 64,
                       'lr': 6.25e-5,
                       'lr_warmup': 16000,
                       'lm_weight': 0.5,
                       'risk_weight': 0,
                       'n_jobs': 4,
                       'label_smoothing': 0.1,
                       'clip_grad': None,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last': True, 
                       'openai_parameters_dir': './parameters',
                       'last_checkpoint_path': './checkpoints/last_checkpoint',
                       'interrupt_checkpoint_path': './checkpoints/interrupt_checkpoint',
                       'train_datasets': ['./datasets/ConvAI2/train_self_revised_no_cands.txt',
                                          './datasets/ConvAI2/train_self_original_no_cands.txt',
                                          './datasets/DailyDialog/train_dailydialog.txt'],
                       'test_datasets': ['./datasets/ConvAI2/valid_self_revised_no_cands.txt',
                                         './datasets/ConvAI2/valid_self_original_no_cands.txt',
                                         './datasets/DailyDialog/valid_dailydialog.txt']})

    return config

