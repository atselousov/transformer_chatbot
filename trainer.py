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
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.model.embeddings.padding_idx)
        base_optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.optimizer = NoamOpt(self.model.embeddings.embedding_dim, 1, lr_warmup, base_optimizer)

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                           num_workers=n_jobs, collate_fn=self.collate_func)
        if test_dataset is not None:
            self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                            num_workers=n_jobs, collate_fn=self.collate_func)

        self.clip_grad = clip_grad
        self.device = device

    def collate_func(self, data):
        data = [d[:self.model.pos_embeddings.num_embeddings - 1] for d in data]
        data = pad_sequence(data, batch_first=True, padding_value=self.model.embeddings.padding_idx)
        return data[:, :-1], data[:, 1:]

    def _train_epoch(self, epoch):
        self.model.train()

        tqdm_data = tqdm(self.train_dataloader, desc='Train (epoch #{})'.format(epoch))
        loss = 0
        for i, (inputs, targets) in enumerate(tqdm_data):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))

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
        for i, (inputs, targets) in enumerate(tqdm_data):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))

            loss = (i * loss + batch_loss.item()) / (i + 1)

            tqdm_data.set_postfix({'loss': loss})

    def train(self, epochs, test_period=1):
        for epoch in range(epochs):
            self._train_epoch(epoch)

            if hasattr(self, 'test_dataloader') and epoch % test_period == 0:
                self._test_epoch(epoch)
