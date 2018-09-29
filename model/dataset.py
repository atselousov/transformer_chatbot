import random
import torch
from torch.utils.data import Dataset
from .text import BPEVocab


class FacebookDataset(Dataset):
    @staticmethod
    def parse_data(path):
        with open(path, 'r', encoding='utf-8') as file:
            data = []
            for line in file.readlines():
                line = line.strip()

                if len(line) == 0:
                    continue

                space_idx = line.find(' ')
                if space_idx == -1:
                    dialog_idx = int(line)
                else:
                    dialog_idx = int(line[:space_idx])

                if int(dialog_idx) == 1:
                    data.append({'persona_info': [], 'dialog': []})

                dialog_line = line[space_idx + 1:].split('\t')
                dialog_line = [l.strip() for l in dialog_line]

                if dialog_line[0].startswith('your persona:'):
                    persona_info = dialog_line[0].replace('your persona: ', '')
                    data[-1]['persona_info'].append(persona_info)

                elif len(dialog_line) > 1:
                    data[-1]['dialog'].append(dialog_line[0])
                    data[-1]['dialog'].append(dialog_line[1])

            return data

    @staticmethod
    def make_dataset(data, vocab, max_lengths):
        dataset = []
        for chat in data:
            persona_info = [vocab.string2ids(s) for s in chat['persona_info']]
            dialog = [vocab.string2ids(s) for s in chat['dialog']]

            if len(dialog) % 2 == 1:
                dialog = dialog[:-1]
           
            dataset.append((persona_info, dialog))

        return dataset

    def __init__(self, paths, vocab, max_lengths=2048, min_infos=2):
        assert min_infos > 0             

        if isinstance(paths, str):
            paths = [paths]
        
        self.vocab = vocab
        self.max_lengths = max_lengths
        self.min_infos = min_infos

        parsed_data = sum([FacebookDataset.parse_data(path) for path in paths], [])
        self.data = FacebookDataset.make_dataset(parsed_data, vocab, max_lengths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        persona_info, dialog = self.data[idx]

        if len(persona_info):
            n_info_samples = max(self.min_infos, random.randint(1, len(persona_info)))
            n_info_samples = min(n_info_samples, len(persona_info))
            persona_info = random.sample(persona_info, n_info_samples)
            random.shuffle(persona_info)
            persona_info = sum(persona_info, []) 
            persona_info = [self.vocab.info_bos_id] + persona_info[:self.max_lengths-2] + [self.vocab.info_eos_id]

        dialog_begin = 0
        dialog_end = random.randrange(2, len(dialog)+1, 2)

        h = []
        for i, ids in enumerate(dialog[dialog_begin:dialog_end-1], 1):
            if i % 2 == 1:
                ids = [self.vocab.talker1_bos_id] + ids + [self.vocab.talker1_eos_id]
            else:
                ids = [self.vocab.talker2_bos_id] + ids + [self.vocab.talker2_eos_id]
            h.extend(ids)
        h = h[-self.max_lengths:]

        y = [self.vocab.bos_id] + dialog[dialog_end-1] + [self.vocab.eos_id]
        y = y[:self.max_lengths]

        return persona_info, h, y
