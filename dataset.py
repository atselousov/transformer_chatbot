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
            persona_info = ' '.join([s for s in dialog['persona_info']])
            persona_info = [vocab.info_bos_id] + vocab.string2ids(persona_info) + [vocab.info_eos_id]

            dialog_x = []
            for i, string in enumerate(dialog['dialog'], 1):
                ids = vocab.string2ids(string)

                if i % 2 == 1:
                    ids = [vocab.talker1_bos_id] + ids + [vocab.talker1_eos_id]
                else:
                    dialog_y = [vocab.bos_id] + ids + [vocab.eos_id]
                    dataset_item = (torch.tensor(persona_info, dtype=torch.long), 
                                    torch.tensor(dialog_x, dtype=torch.long),
                                    torch.tensor(dialog_y, dtype=torch.long))
                    dataset.append(dataset_item)

                    ids = [vocab.talker2_bos_id] + ids + [vocab.talker2_eos_id]

                dialog_x.extend(ids)

        return dataset

    def __init__(self, paths, vocab):
        if isinstance(paths, str):
            paths = [paths]
        
        self.data = sum([FacebookDataset.parse_data(path) for path in paths], [])
        self.data = FacebookDataset.make_dataset(self.data, vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        persona_info, x, y = self.data[idx]
        return persona_info, x, y
