import torch
import torch.nn as nn

from transformer import TransformerModule

class TransformerModel(nn.Module):
    def __init__(self, n_layers, n_embeddings, n_pos_embeddings, embeddings_size, 
                 padding_idx, n_heads, dropout, embed_dropout, attn_dropout, ff_dropout, 
                 n_segments=None):

        super(TransformerModel, self).__init__()

        self.padding_idx = padding_idx
        self.n_pos_embeddings = n_pos_embeddings
        self.embeddings_size = embeddings_size

        self.transformer_module = TransformerModule(n_layers, n_embeddings, n_pos_embeddings, embeddings_size, 
                                                    padding_idx, n_heads, dropout, embed_dropout, attn_dropout,
                                                    ff_dropout, n_segments)
        self.pre_softmax = nn.Linear(embeddings_size, n_embeddings, bias=False)
        self.pre_softmax.weight = self.transformer_module.embeddings.weight

    '''def length_penalty(self, sequence_lengths):
        """https://arxiv.org/abs/1609.08144"""
        return (5 + sequence_lengths) ** self.opt['length_penalty_coef'] / (5 + 1) ** self.opt['length_penalty_coef']

    def beam_search(self, mem, lens, max_len, sampling=False):
        with torch.no_grad():
            beam_size = self.opt['n_beams']
            batch_size = lens.shape[0]

            prevs = torch.empty(1, batch_size * beam_size, 1, dtype=torch.long, device=mem.device).fill_(self.dict[OnmtAgent.BOS])
            
            beam_scores = torch.zeros(batch_size, beam_size, device=mem.device)
            beam_lens = torch.ones(batch_size, beam_size, dtype=torch.long, device=mem.device)
            is_end = torch.zeros(batch_size, beam_size, dtype=torch.uint8, device=mem.device)
            
            mem = mem.repeat(1, beam_size, 1)
            lens = lens.view(-1, 1).repeat(1, beam_size).view(-1)

            decoder_state = self.decoder.init_decoder_state(None, mem, None)
            lm_mem = torch.zeros_like(mem, device=mem.device)
            lm_decoder_state = self.decoder.init_decoder_state(None, lm_mem, None)

            antilm_n = 5
            for i in range(max_len):
                outputs, decoder_state, _ =  self.decoder(prevs, mem, decoder_state, lens)
                probs = self.generator(outputs)
                probs = probs.view(batch_size, beam_size, -1)

                if self.opt['anti_lm_coef'] > 0 and i < antilm_n:
                    lm_outputs, lm_decoder_state, _ =  self.decoder(prevs, lm_mem, lm_decoder_state, lens)
                    lm_probs = self.generator(lm_outputs)
                    lm_probs = lm_probs.view(batch_size, beam_size, -1)
                    probs -= self.opt['anti_lm_coef'] * lm_probs

                beam_scores = beam_scores.repeat(1, probs.shape[-1]) + probs.view(batch_size, -1) * (1 - is_end.repeat(1, probs.shape[-1]).float())

                prenalty = self.length_penalty(beam_lens.float() + 1 - is_end.float()).repeat(1, probs.shape[-1])
                beam_scores = beam_scores / prenalty

                if sampling:
                    assert False
                    #currents = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(-1, batch_size)
                else:
                    beam_scores, idxs = beam_scores.topk(beam_size, dim=-1)
                

                beam_idxs = (idxs.float() / probs.shape[-1]).long()            
                sym_idxs = torch.fmod(idxs, probs.shape[-1])
                
                beam_scores *= torch.gather(prenalty, 1, beam_idxs)
               
                is_end = torch.gather(is_end, 1, beam_idxs)
                is_end[sym_idxs == self.dict[OnmtAgent.EOS]] = 1
                if all(is_end.view(-1)):
                    break

                beam_lens = torch.gather(beam_lens, 1, beam_idxs)
                beam_lens[~is_end] += 1
                
                decoder_state.previous_input = decoder_state.previous_input[:, beam_idxs.view(-1), :]
                decoder_state.previous_layer_inputs = decoder_state.previous_layer_inputs[:, beam_idxs.view(-1), :, :]
                if self.opt['anti_lm_coef'] > 0 and i < antilm_n:
                    lm_decoder_state.previous_input = lm_decoder_state.previous_input[:, beam_idxs.view(-1), :]
                    lm_decoder_state.previous_layer_inputs = lm_decoder_state.previous_layer_inputs[:, beam_idxs.view(-1), :, :]
                
                prevs = sym_idxs.view(1, batch_size * beam_size, 1)

            predicts = []
            result = decoder_state.previous_input.view(-1, batch_size, beam_size)
            bests = beam_scores.argmax(dim=-1)
            for i in range(batch_size):
                try:
                    best_len = beam_lens[i, bests[i]]
                    best_seq = result[1:best_len, i, bests[i]]
                    predicts.append(best_seq.tolist())
                except Exception:
                    predicts.appen([])
                
        return predicts

    def _predict_seq(self, valid_observations, max_len=50, sampling=False):
        self.eval()
        
        batch_size = len(valid_observations)

        hists = [obs['agent'].history['info'] + list(obs['agent'].history['dialog']) for obs in valid_observations]
        hists = [self._compute_mem(hist) for hist in hists]
        
        mem = pad_sequence(hists)
        lens = torch.tensor([h.shape[0] for h in hists], dtype=torch.long, device=mem.device)

        try:
            predicts = self.beam_search(mem, lens, max_len)
        except Exception:
            predicts = [[] for i in range(batch_size)]
        
        return predicts

    '''

    def forward(self, x, contexts=[]):
        enc_contexts = [self.encode(c) for c in contexts]
        return self.decode(x, enc_contexts)

    def encode(self, x):
        return self.transformer_module(x)

    def decode(self, x, enc_contexts=[], only_last=False):
        x, _ = self.transformer_module(x, enc_contexts)

        if only_last:
            x = x[:, -1, :]

        return self.pre_softmax(x)

    def predict(self):
        pass
