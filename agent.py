import torch
import torch.nn.functional as F
from collections import deque
from parlai.core.agents import Agent
from model.transformer_model import TransformerModel
from model.text import BPEVocab
from model.utils import pad_sequence
from model.postprocessing import ngram_replaser, ReplyChecker, detokenize, syntax_fix
from model.retrieval import RetrievalBot, DIALOG_SIZE
from model.sentiment import pick_emoji, clean_emoji
from config import get_model_config
import random

      
class TransformerAgent(Agent):
    @staticmethod
    def add_cmdline_args(argparser):
        agent_args = argparser.add_argument_group('Agent parameters')
        agent_args.add_argument('-gpu', '--gpu', type=int, default=-1, 
                                help='which GPU to use')
        agent_args.add_argument('--no-cuda', type='bool', default=False,
                                help='disable GPUs even if available. otherwise, will use GPUs if '
                                     'available on the device.')
        agent_args.add_argument('--rank_candidates', type='bool', default=False,
                                help='Whether the model should parse candidates for ranking.')
        agent_args.add_argument('--sample', type='bool', default=True,
                                help='Sampling of beam from beam search')
        
        return argparser

    def __init__(self, opt, shared=None):
        super(TransformerAgent, self).__init__(opt, shared)

        self.use_cuda = not self.opt.get('no_cuda') and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(self.opt['gpu'])

        torch.set_grad_enabled(False)

        model_config = get_model_config()
        self.vocab = BPEVocab.from_files(model_config.bpe_vocab_path, model_config.bpe_codes_path)
        self.reply_checker = ReplyChecker(correct_generative=model_config.correct_generative,
                                          split_into_sentences=model_config.split_into_sentences)

        self.replace_repeat = model_config.replace_repeat
        self.replace_ngram = model_config.replace_ngram
        self.ngram_size = model_config.ngram_size
        self.detokenize = model_config.detokenize
        self.emoji_prob = model_config.emoji_prob
        self.add_questions = model_config.add_questions
        self.beam_size = model_config.beam_size

        self.clean_emoji = model_config.clean_emoji
        self.check_grammar = model_config.check_grammar

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
                                          sample=self.opt['sample'],
                                          annealing=model_config.annealing)
            self.retrieval_bot = RetrievalBot()

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
            self.retrieval_bot = shared['retrieval']

        self.reset()

    def _preprocess_text(self, text):
        # print('Original text: ', text)
        if self.clean_emoji:
            text = clean_emoji(text)
            # print('After emoji clean: ', text)

        if self.check_grammar:
            text = syntax_fix(text).lower()
            # print('After syntax fix: ', text)

        return text

    def _parse(self, text):
        # todo: fix grammar mistakes?
        persona_info = []
        dialog = []
        for subtext in text.split('\n'):
            subtext = subtext.strip()
            if subtext.startswith('your persona:'):
                subtext = subtext.replace('your persona:', '').strip()
                subtext = self._preprocess_text(subtext).strip()
                persona_info.append(subtext)
            else:
                subtext = self._preprocess_text(subtext).strip()
                dialog.append(subtext)

        return persona_info, dialog

    def observe(self, observation):
        if self.episode_done:
            self.reset()

        if 'text' in observation:
            text = observation['text']
            info, dialog = self._parse(text)

            if info:
                self.history['str_info'] = ' '.join(info)
            self.history['str_dialog'].extend(dialog)
        
            info = sum([self.vocab.string2ids(i) for i in info], [])
            self.history['info'].extend(info)

            for i, d in enumerate(dialog, 1):
                d = self.vocab.string2ids(d)
                if i % 2 == 1:
                    d = [self.vocab.talker1_bos_id] + d + [self.vocab.talker1_eos_id]
                else:
                    d = [self.vocab.talker2_bos_id] + d + [self.vocab.talker2_eos_id]

                self.history['dialog'].extend(d)

        observation['agent'] = self        

        self.episode_done = observation['episode_done']
        self.observation = observation
        
        return observation
    
    def act(self):
        return self.batch_act([self.observation])[0]

    def _postprocess_text(self, reply, agent):
        str_reply = self.vocab.ids2string(reply)

        # print('Original reply: ', str_reply)
        if self.replace_repeat:
            str_reply = agent.reply_checker.check_reply(str_reply,
                                                        agent.history['str_dialog'][-1],
                                                        agent.history['str_info'])
            # print('After repeat replace: ', str_reply)

        if self.beam_size > 1 and random.uniform(0, 1) < self.add_questions and '?' not in str_reply:
            question = self.retrieval_bot.generate_question(list(agent.history['str_dialog']),
                                                            agent.history['str_info'])
            if question is not None and question not in str_reply:
                str_reply = ' '.join([str_reply, question])

        # print('Question: ', question)

        if self.replace_ngram:
            str_reply = ngram_replaser(agent.history['str_info'], str_reply, n=self.ngram_size)
            # print('After ngram replace: ', str_reply)

        reply = self.vocab.string2ids(str_reply)

        if self.detokenize:
            str_reply = detokenize(str_reply)
            # print('After detokenize: ', str_reply)

        if random.uniform(0, 1) < self.emoji_prob:
            str_reply = ' '.join([str_reply, pick_emoji(str_reply)])

        return str_reply, reply

    def batch_act(self, observations):
        def is_valid_history(history):
            return len(history['dialog'])

        def to_tensor(string):
            ids = [self.vocab.bos_id] + self.vocab.string2ids(string) + [self.vocab.eos_id]
            return torch.tensor(ids, dtype=torch.long)

        batch_reply = [{'id': self.getID(), 'text': '', 'text_candidates': []} for _ in range(len(observations))]
        valid_ids = [i for i, obs in enumerate(observations) if is_valid_history(obs['agent'].history)]
        batch_size = len(valid_ids)

        if batch_size == 0:
            return batch_reply

        try:
            valid_observations = [observations[i] for i in valid_ids]

            infos = [obs['agent'].history['info'][:self.model.n_pos_embeddings-3] for obs in valid_observations]
            infos = [([self.vocab.info_bos_id] + ifo + [self.vocab.info_eos_id] if len(ifo) else ifo) for ifo in infos]
            dialogs = [list(obs['agent'].history['dialog'])[-self.model.n_pos_embeddings+1:] for obs in valid_observations]
            contexts = []

            if max(map(len, infos)) > 0:
                infos = [torch.tensor(i, dtype=torch.long) for i in infos]
                infos = pad_sequence(infos, batch_first=True, padding_value=self.model.padding_idx)
                if self.use_cuda:
                    infos = infos.cuda()
                contexts.append(infos)

            if max(map(len, dialogs)) > 0:
                dialogs = [torch.tensor(d, dtype=torch.long) for d in dialogs]
                dialogs = pad_sequence(dialogs, batch_first=True, padding_value=self.model.padding_idx)
                if self.use_cuda:
                    dialogs = dialogs.cuda()
                contexts.append(dialogs)

            enc_contexts = [self.model.encode(c) for c in contexts]
            pred_texts = self.model.beam_search(enc_contexts)

            for i in range(batch_size):
                pred_text_str, pred_text = self._postprocess_text(pred_texts[i], valid_observations[i]['agent'])

                valid_observations[i]['agent'].history['dialog'].extend([self.vocab.talker2_bos_id] + pred_text)
                batch_reply[valid_ids[i]]['text'] = pred_text_str

            if self.opt['rank_candidates']:
                candidates = [list(obs.get('label_candidates', [])) for obs in valid_observations]
                lens_candidates = [len(c) for c in candidates]

                if max(lens_candidates) > 0:
                    candidates = [c + ['' for _ in range(max(lens_candidates) - len(c))] for c in candidates]
                    scores = [[] for _ in range(len(candidates))]

                    for i in range(max(lens_candidates)):
                        current_cands = [to_tensor(c[i])[:self.model.n_pos_embeddings-1] for c in candidates]
                        current_cands = pad_sequence(current_cands, batch_first=True, padding_value=self.model.padding_idx)
                        if self.use_cuda:
                            current_cands = current_cands.cuda()

                        logits = self.model.decode(current_cands[:, :-1], enc_contexts)
                        log_probas = F.log_softmax(logits, dim=-1)
                        log_probas = torch.gather(log_probas, -1, current_cands[:, 1:].unsqueeze(-1)).squeeze(-1)
                        log_probas.masked_fill_(current_cands[:, 1:].eq(self.model.padding_idx), 0)

                        current_lens = current_cands[:, 1:].ne(self.model.padding_idx).float().sum(dim=-1)
                        current_scores = log_probas.sum(dim=-1) / current_lens

                        for k, s in enumerate(current_scores):
                            if i < lens_candidates[k]:
                                scores[k].append(s.item())

                    ranked_ids = [sorted(range(len(s)), key=lambda k: s[k], reverse=True) for s in scores]
                    ranked_strings = [[c[i] for i in ids] for ids, c in zip(ranked_ids, candidates)]

                    for i in range(batch_size):
                        batch_reply[valid_ids[i]]['text_candidates'] = ranked_strings[i]

        except Exception as e:
            raise e
            print(e)

        return batch_reply

    def share(self):
        shared = super(TransformerAgent, self).share()
        shared['opt'] = self.opt
        shared['model'] = self.model
        shared['retrieval'] = self.retrieval_bot

        return shared

    def reset(self):
        self.history = {'str_info': None, 'str_dialog': deque(DIALOG_SIZE * ['None'], maxlen=DIALOG_SIZE),
                        'info': [], 'dialog': deque(maxlen=self.model.n_pos_embeddings-1)}
        self.episode_done = True
        self.observation = None
        self.reply_checker.clean()

