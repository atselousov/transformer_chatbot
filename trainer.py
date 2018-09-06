import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import pad_sequence
from optim import Adam, NoamOpt
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_dataset, test_dataset=None, batch_size=8,
                 batch_split=1, lm_weight=0.5, lr=6.25e-5, lr_warmup=2000, n_jobs=0, 
                 clip_grad=1, device=torch.device('cuda')):
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.model.padding_idx)
        base_optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.optimizer = NoamOpt(self.model.embeddings_size, 1, lr_warmup, base_optimizer)

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size//batch_split, shuffle=True, 
                                           num_workers=n_jobs, collate_fn=self.collate_func)
        if test_dataset is not None:
            self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size//batch_split, shuffle=False, 
                                            num_workers=n_jobs, collate_fn=self.collate_func)

        self.batch_split = batch_split
        self.lm_weight = lm_weight
        self.clip_grad = clip_grad
        self.device = device

    def collate_func(self, data):
        persona_info, x, y = zip(*data)

        persona_info = [t[:self.model.n_pos_embeddings - 1] for t in persona_info]
        persona_info = pad_sequence(persona_info, batch_first=True, padding_value=self.model.padding_idx)
        x = [t[-self.model.n_pos_embeddings + 1:] for t in x]
        x = pad_sequence(x, batch_first=True, padding_value=self.model.padding_idx)
        y = [t[:self.model.n_pos_embeddings - 1] for t in y]
        y = pad_sequence(y, batch_first=True, padding_value=self.model.padding_idx)

        return [x, persona_info], y

    def _eval_train(self, epoch):
        self.model.train()

        tqdm_data = tqdm(self.train_dataloader, desc='Train (epoch #{})'.format(epoch))
        loss = 0
        lm_loss = 0
        for i, (contexts, targets) in enumerate(tqdm_data):
            contexts, targets = [c.to(self.device) for c in contexts], targets.to(self.device)
            
            enc_contexts = []
            batch_lm_loss = 0
            for context in contexts:
                enc_context = self.model.encode(context)
                context_outputs = self.model.generate(enc_context[0])
                prevs, nexts = context_outputs[:, :-1, :].contiguous(), context[:, 1:].contiguous()
                batch_lm_loss += self.criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1))             

                enc_contexts.append(enc_context)
            
            batch_lm_loss /= len(contexts)

            prevs, nexts = targets[:, :-1].contiguous(), targets[:, 1:].contiguous()
            outputs = self.model.decode(prevs, enc_contexts)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))
            
            full_loss = (batch_lm_loss * self.lm_weight + batch_loss) / self.batch_split
            full_loss.backward()
            
            if (i + 1) % self.batch_split == 0:
                if self.clip_grad is not None:
                    for group in self.optimizer.param_groups:
                        nn.utils.clip_grad_norm_(group['params'], self.clip_grad)

                self.optimizer.step()
                self.optimizer.zero_grad()

            lm_loss = (i * lm_loss + batch_lm_loss.item()) / (i + 1)
            loss = (i * loss + batch_loss.item()) / (i + 1)

            tqdm_data.set_postfix({'lm_loss': lm_loss, 'loss': loss})

    def _eval_test(self, metric_funcs={}):
        self.model.eval()

        tqdm_data = tqdm(self.test_dataloader, desc='Test')
        loss = 0
        lm_loss = 0
        metrics = {name: 0 for name in metric_funcs.keys()}
        for i, (contexts, targets) in enumerate(tqdm_data):
            contexts, targets = [c.to(self.device) for c in contexts], targets.to(self.device)
            
            enc_contexts = []
            batch_lm_loss = 0
            for context in contexts:
                enc_context = self.model.encode(context)
                context_outputs = self.model.generate(enc_context[0])
                prevs, nexts = context_outputs[:, :-1, :].contiguous(), context[:, 1:].contiguous()
                batch_lm_loss += self.criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1))             

                enc_contexts.append(enc_context)
            
            batch_lm_loss /= len(contexts)

            prevs, nexts = targets[:, :-1].contiguous(), targets[:, 1:].contiguous()
            outputs = self.model.decode(prevs, enc_contexts)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))
            
            predictions = self.model.beam_search(enc_contexts)
            target_lens = targets.ne(self.model.padding_idx).sum(dim=-1)
            targets = [t[1:l-1].tolist() for t, l in zip(targets, target_lens)]
            
            lm_loss = (i * lm_loss + batch_lm_loss.item()) / (i + 1)
            loss = (i * loss + batch_loss.item()) / (i + 1)
            for name, func in metric_funcs.items():
                score = func(predictions, targets)
                metrics[name] = (metrics[name] * i + score) / (i + 1)

            tqdm_data.set_postfix(dict({'lm_loss': lm_loss, 'loss': loss}, **metrics))
    
    def test(self, metric_funcs={}):
        if hasattr(self, 'test_dataloader'):
            self._eval_test(metric_funcs)

    def train(self, epochs, after_epoch_funcs=[]):
        for epoch in range(epochs):
            self._eval_train(epoch)

            for func in after_epoch_funcs:
                func(epoch)
