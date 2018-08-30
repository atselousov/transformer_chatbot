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

                else:
                    data[-1]['dialog'].append(dialog_line[0])
                    data[-1]['dialog'].append(dialog_line[1])

            return data

    @staticmethod
    def make_dataset(data, vocab):
        dataset = []
        for dialog in data:
            strings = [s for s in dialog['persona_info']] + [s for s in dialog['dialog']]
            ids_list = [vocab.string2ids(s, add_bos=True, add_eos=True) for s in strings]
            ids_list = [torch.tensor(ids, dtype=torch.long) for ids in ids_list]
            dataset.extend(ids_list)

        return dataset

    def __init__(self, path, vocab):
        self.data = FacebookDataset.parse_data(path)
        self.data = FacebookDataset.make_dataset(self.data, vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]