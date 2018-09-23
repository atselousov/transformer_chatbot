import torch
from torch.utils.data import Dataset
from text import BPEVocab


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
        for dialog in data:
            persona_info = ' '.join([s for s in dialog['persona_info']])
            persona_info = vocab.string2ids(persona_info)

            dialog_x = []
            for i, string in enumerate(dialog['dialog'], 1):
                ids = vocab.string2ids(string)

                if i % 2 == 1:
                    ids = [vocab.talker1_bos_id] + ids
                else:
                    dialog_y = [vocab.bos_id] + ids + [vocab.eos_id]
                    dataset_item = (persona_info[:max_lengths],
                                    dialog_x[-max_lengths:],
                                    dialog_y[:max_lengths])
                    dataset.append(dataset_item)

                    ids = [vocab.talker2_bos_id] + ids

                dialog_x.extend(ids)
                
        split_size = 10000
        n = len(dataset) // split_size
        dataset = [dataset[i::n] for i in range(n)]
        dataset = [sorted(d, key=lambda x: (len(x[1]), len(x[0]), len(x[2]))) for d in dataset]
        dataset = sum(dataset, [])

        return dataset

    def __init__(self, paths, vocab, max_lengths=2048):
        if isinstance(paths, str):
            paths = [paths]
        
        self.data = sum([FacebookDataset.parse_data(path) for path in paths], [])
        self.data = FacebookDataset.make_dataset(self.data, vocab, max_lengths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        persona_info, h, y = self.data[idx]
        return persona_info, h, y
