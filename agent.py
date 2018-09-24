import torch
from collections import deque
from parlai.core.agents import Agent
from model.transformer_model import TransformerModel
from model.text import BPEVocab
from model.utils import pad_sequence
from config import get_model_config

      
class TransformerAgent(Agent):
    @staticmethod
    def add_cmdline_args(argparser):
        agent_args = argparser.add_argument_group('Agent parameters')
        agent_args.add_argument('-gpu', '--gpu', type=int, default=-1, 
                                help='which GPU to use')
        agent_args.add_argument('--no-cuda', type='bool', default=False,
                                help='disable GPUs even if available. otherwise, will use GPUs if '
                                     'available on the device.')
        
        return argparser

    def __init__(self, opt, shared=None):
        super(TransformerAgent, self).__init__(opt, shared)
        
        self.use_cuda = not self.opt.get('no_cuda') and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(self.opt['gpu'])

        model_config = get_model_config()
        self.vocab = BPEVocab.from_files(model_config.bpe_vocab_path, model_config.bpe_codes_path)

        if shared is None:
            self.model = TransformerModel(n_layers=model_config.n_layers,
                                          n_embeddings=len(self.vocab),
                                          n_pos_embeddings=model_config.n_pos_embeddings,
                                          embeddings_size=model_config.embeddings_size,
                                          padding_idx=self.vocab.pad_id,
                                          n_heads=model_config.n_heads,
                                          dropout=model_config.dropout,
                                          embed_dropout=model_config.embed_dropout,
                                          attn_dropout=model_config.attn_dropout,
                                          ff_dropout=model_config.ff_dropout,
                                          bos_id=self.vocab.bos_id,
                                          eos_id=self.vocab.eos_id,
                                          max_seq_len=model_config.max_seq_len,
                                          beam_size=model_config.beam_size,  
                                          length_penalty=model_config.length_penalty,
                                          n_segments=model_config.n_segments,
                                          sample=model_config.sample)

            state_dict = torch.load(model_config.checkpoint_path, map_location=lambda storage, loc: storage)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self.model.load_state_dict(state_dict)
            print('Weights loaded from {}'.format(model_config.checkpoint_path))

            if self.use_cuda:
                self.model = self.model.cuda()

            self.model.eval()

        else:
            self.model = shared['model']

        self.reset()

    def _parse(self, text):
        persona_info = []
        dialog = []
        for subtext in text.split('\n'):
            if subtext.startswith('your persona:'):
                subtext = sentence.replace('your persona:', '').strip()
                persona_info.append(subtext)
            else:
                dialog.append(subtext)

        return persona_info, dialog

    def observe(self, observation):
        if self.episode_done:
            self.reset()

        if 'text' in observation:
            text = observation['text']
            info, dialog = self._parse(text)

            info = [self.vocab.string2ids(i) for i in info]
            dialog = [[self.vocab.talker1_bos_id if k % 2 == 1 else self.vocab.talker2_bos_id] + \
                      self.vocab.string2ids(d) for k, d in enumerate(dialog, 1)]

            self.history['info'].extend(sum(info, []))
            self.history['dialog'].extend(sum(dialog, []))

        observation['agent'] = self        

        self.episode_done = observation['episode_done']
        self.observation = observation

        return observation
    
    def act(self):
        return self.batch_act([self.observation])[0]       

    def batch_act(self, observations):
        batch_reply = [{'id': self.getID(), 'text': ''} for _ in range(len(observations))]

        valid_ids = [i for i, obs in enumerate(observations) 
                        if (len(obs['agent'].history['info']) or len(obs['agent'].history['dialog']))]
        batch_size = len(valid_ids)

        if batch_size != 0:
            valid_observations = [observations[i] for i in valid_ids]

            info = [obs['agent'].history['info'][:self.model.n_pos_embeddings-1] for obs in valid_observations]
            dialog = [obs['agent'].history['dialog'][-self.model.n_pos_embeddings+1:] for obs in valid_observations]

            contexts = []

            if max(map(len, info)) > 0:
                info = [torch.tensor(d, dtype=torch.long) for d in info]
                info = pad_sequence(info, batch_first=True, padding_value=self.model.padding_idx)
                contexts.append(info)

            if max(map(len, dialog)) > 0:
                dialog = [torch.tensor(d, dtype=torch.long) for d in dialog]
                dialog = pad_sequence(dialog, batch_first=True, padding_value=self.model.padding_idx)
                contexts.append(dialog)

            pred_texts = self.model.predict(contexts)
            for i in range(batch_size):
                pred_text = pred_texts[i]
                valid_observations[i]['agent'].history['dialog'].extend([self.vocab.talker2_bos_id] + pred_text)
                pred_text =  self.vocab.ids2string(pred_text)
                batch_reply[valid_ids[i]]['text'] = pred_text

        else:
            print('Not valid batch!')

        return batch_reply

    def share(self):
        shared = super(TransformerAgent, self).share()
        shared['opt'] = self.opt
        shared['model'] = self.model

        return shared

    def reset(self):
        self.history = {'info': [], 'dialog': deque(maxlen=self.model.n_pos_embeddings-1))}
        self.episode_done = True
        self.observation = None

