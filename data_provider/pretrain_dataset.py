import torch
from torch_geometric.data import Dataset
import os
import random
from pathlib import Path

class GINPretrainDataset(Dataset):
    def __init__(self, root, text_max_len, graph_aug, text_aug, filtered_cid_path=None):
        super(GINPretrainDataset, self).__init__(root)
        self.root = root
        self.graph_aug = graph_aug
        self.text_aug = text_aug
        self.text_max_len = text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.tokenizer = None
        if filtered_cid_path is not None:
            print('before filtering', len(self.graph_name_list), len(self.text_name_list))
            with open(filtered_cid_path, 'r') as f:
                self.filtered_cid_set = [line.strip() for line in f.readlines()]
                self.filtered_cid_set = set(self.filtered_cid_set)
            filtered_graph_name_list = []
            for g in self.graph_name_list:
                g = Path(g)
                cid = g.name.split('_')[1][:-3]
                if cid in self.filtered_cid_set:
                    filtered_graph_name_list.append(g)
            self.graph_name_list = filtered_graph_name_list
            filtered_text_name_list = []
            for t in self.text_name_list:
                t = Path(t)
                cid = t.name.split('_')[1][:-4]
                if cid in self.filtered_cid_set:
                    filtered_text_name_list.append(t)
            self.text_name_list = filtered_text_name_list
            print('after filtering', len(self.graph_name_list), len(self.text_name_list))
            

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        # load and process graph
        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)
        # load and process text
        text_path = os.path.join(self.root, 'text', text_name)
        if self.text_aug:
            text_list = []
            count = 0
            for line in open(text_path, 'r', encoding='utf-8'):
                count += 1
                text_list.append(line.strip('\n') + '\n')
                if count > 100:
                    break
            text_sample = random.sample(text_list, 1)
            text_list.clear()
            text, mask = self.tokenizer_text(text_sample[0])
        else:
            with open(text_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines if line.strip()]
            text = ' '.join(lines) + '\n'
            text, mask = self.tokenizer_text(text)
        return data_graph, text.squeeze(0), mask.squeeze(0)


    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask

