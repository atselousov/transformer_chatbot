import torch
import torch.nn as nn

from transformer import TransformerModule

class TransformerModel(nn.Module):
    def __init__(self, n_layers, n_embeddings, n_pos_embeddings, embeddings_size, 
                 padding_idx, n_heads, dropout, embed_dropout, attn_dropout, ff_dropout,
                 bos_id, eos_id, max_seq_len=256, beam_size=5, sample=False, 
                 length_penalty=0.8, n_segments=None):

        super(TransformerModel, self).__init__()

        self.padding_idx = padding_idx
        self.n_pos_embeddings = n_pos_embeddings
        self.embeddings_size = embeddings_size

        self.bos_id = bos_id
        self.eos_id = eos_id

        self.max_seq_len = max_seq_len
        self.beam_size = beam_size
        self.sample = sample
        self.length_penalty_coef = length_penalty

        self.transformer_module = TransformerModule(n_layers, n_embeddings, n_pos_embeddings, embeddings_size, 
                                                    padding_idx, n_heads, dropout, embed_dropout, attn_dropout,
                                                    ff_dropout, n_segments)
        self.pre_softmax = nn.Linear(embeddings_size, n_embeddings, bias=False)
        self.pre_softmax.weight = self.transformer_module.embeddings.weight

    def forward(self, x, contexts=[]):
        enc_contexts = [self.encode(c) for c in contexts]
        return self.decode(x, enc_contexts)

    def encode(self, x):
        return self.transformer_module(x)

    def generate(self, enc_x):
        return self.pre_softmax(enc_x)

    def decode(self, x, enc_contexts=[]):
        x, _ = self.transformer_module(x, enc_contexts)
        return self.generate(x)

    def predict(self, contexts=[]):
        enc_contexts = [encode(c) for c in contexts]
        return self.beam_search(enc_contexts)

    def _length_penalty(self, sequence_lengths):
        """https://arxiv.org/abs/1609.08144"""
        return (5 + sequence_lengths) ** self.length_penalty_coef / (5 + 1) ** self.length_penalty_coef

    def beam_search(self, enc_contexts=[]):
        with torch.no_grad():
            if len(enc_contexts) == 0:
                return []

            batch_size = enc_contexts[0][0].shape[0]
            device = next(self.parameters()).device

            prevs = torch.full((batch_size * self.beam_size, 1), fill_value=self.bos_id, dtype=torch.long, device=device)
            
            beam_scores = torch.zeros(batch_size, self.beam_size, device=device)
            beam_lens = torch.ones(batch_size, self.beam_size, dtype=torch.long, device=device)
            is_end = torch.zeros(batch_size, self.beam_size, dtype=torch.uint8, device=device)
            
            enc_contexts = [(c.repeat(self.beam_size, 1, 1), p.repeat(self.beam_size, 1)) for c, p in enc_contexts]

            for i in range(self.max_seq_len):
                outputs, _ = self.transformer_module(prevs, enc_contexts)

                probs = self.pre_softmax(outputs[:, -1, :])
                probs = probs.view(batch_size, self.beam_size, -1)

                beam_scores = beam_scores.repeat(1, probs.shape[-1]) + probs.view(batch_size, -1) * (1 - is_end.repeat(1, probs.shape[-1]).float())
                prenalty = self._length_penalty(beam_lens.float() + 1 - is_end.float()).repeat(1, probs.shape[-1])
                beam_scores = beam_scores / prenalty
                beam_scores, idxs = beam_scores.topk(self.beam_size, dim=-1)
                
                beam_idxs = (idxs.float() / probs.shape[-1]).long()            
                sym_idxs = torch.fmod(idxs, probs.shape[-1])
                
                beam_scores *= torch.gather(prenalty, 1, beam_idxs)
               
                is_end = torch.gather(is_end, 1, beam_idxs)
                is_end[sym_idxs == self.eos_id] = 1
                if all(is_end.view(-1)):
                    break

                beam_lens = torch.gather(beam_lens, 1, beam_idxs)
                beam_lens[~is_end] += 1
                
                sym_idxs = sym_idxs.view(batch_size * self.beam_size, 1)

                prevs = torch.cat([prevs, sym_idxs], dim=1)

            predicts = []
            result = prevs.view(batch_size, self.beam_size, -1)

            bests = torch.randint(0, self.beam_size, batch_size) if self.sample else beam_scores.argmax(dim=-1)
            
            for i in range(batch_size):
                best_len = beam_lens[i, bests[i]]
                best_seq = result[i, bests[i], 1:best_len]
                predicts.append(best_seq.tolist())
                
        return predicts
