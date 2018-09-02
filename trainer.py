import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import pad_sequence
from optim import Adam, NoamOpt
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_dataset, test_dataset=None, 
                 batch_size=8, lr=6.25e-5, lr_warmup=2000, n_jobs=0, 
                 clip_grad=1, device=torch.device('cuda')):
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.model.padding_idx)
        base_optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.optimizer = NoamOpt(self.model.embeddings_size, 1, lr_warmup, base_optimizer)

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                           num_workers=n_jobs, collate_fn=self.collate_func)
        if test_dataset is not None:
            self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                            num_workers=n_jobs, collate_fn=self.collate_func)

        self.clip_grad = clip_grad
        self.device = device

    def collate_func(self, data):
        persona_info, x, y = zip(*data)

        persona_info = [t[:self.model.n_pos_embeddings] for t in persona_info]
        persona_info = pad_sequence(persona_info, batch_first=True, padding_value=self.model.padding_idx)
        x = [t[-self.model.n_pos_embeddings:] for t in x]
        x = pad_sequence(x, batch_first=True, padding_value=self.model.padding_idx)
        y = [t[:self.model.n_pos_embeddings] for t in y]
        y = pad_sequence(y, batch_first=True, padding_value=self.model.padding_idx)

        return [x, persona_info], y

    def _train_epoch(self, epoch):
        self.model.train()

        tqdm_data = tqdm(self.train_dataloader, desc='Train (epoch #{})'.format(epoch))
        loss = 0
        for i, (contexts, targets) in enumerate(tqdm_data):
            contexts, targets = [c.to(self.device) for c in contexts], targets.to(self.device)
            prev_targets, next_targets = targets[:, :-1].contiguous(), targets[:, 1:].contiguous()
            outputs = self.model(prev_targets, contexts)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), next_targets.view(-1))

            self.optimizer.zero_grad()
            batch_loss.backward()

            if self.clip_grad is not None:
                for group in self.optimizer.param_groups:
                    nn.utils.clip_grad_norm_(group['params'], self.clip_grad)

            self.optimizer.step()

            loss = (i * loss + batch_loss.item()) / (i + 1)

            tqdm_data.set_postfix({'loss': loss})

    def _test_epoch(self, epoch):
        self.model.test()

        tqdm_data = tqdm(self.test_dataloader, desc='Test (epoch #{})'.format(epoch))
        loss = 0
        for i, (contexts, targets) in enumerate(tqdm_data):
            contexts, targets = [c.to(self.device) for c in contexts], targets.to(self.device)
            prev_targets, next_targets = targets[:, :-1], targets[:, 1:]
            outputs = self.model(prev_targets, contexts)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), next_targets.view(-1))

            loss = (i * loss + batch_loss.item()) / (i + 1)

            tqdm_data.set_postfix({'loss': loss})

    def train(self, epochs, test_period=1):
        for epoch in range(epochs):
            self._train_epoch(epoch)

            if hasattr(self, 'test_dataloader') and epoch % test_period == 0:
                self._test_epoch(epoch)
