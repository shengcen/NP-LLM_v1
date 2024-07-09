import torch
from torch.utils.data import DataLoader

from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import InMemoryDataset


class myown(InMemoryDataset):
    def __init__(self):
        super(myown, self).__init__()

    def process_save(self,my_data_list):
        data_list = my_data_list
        data, slices = self.collate(data_list)

        torch.save((data, slices), "./pretrain.pt")
        torch.save((data, slices), "./train.pt")
        torch.save((data, slices), "./valid.pt")
        torch.save((data, slices), "./test.pt")











training_data = []
# 由于是无向图，因此有 4 条边：(0 -> 1), (1 -> 0), (1 -> 2), (2 -> 1)

# 节点的特征

# x = torch.tensor([[5,0,4,5,3,0,2,0,0], [5,0,4,5,3,0,2,0,0], [5,0,4,5,3,0,2,0,0]], dtype=torch.long)
x = torch.tensor([[20,0,1,2,4,5,2,3,1], [60,1,4,5,3,0,2,0,0], [100,2,4,5,3,0,2,0,0]], dtype=torch.long)
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
edge_attr = torch.tensor([[0,0,1], [0,0,1], [0,0,1], [0,0,1]], dtype=torch.long)
smiles = ''
text = "One."
# batch = torch.tensor([0,0,0,1,1,1], dtype=torch.float)

training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,2,2]))
training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,2,2]))
training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,2,2]))
training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,2,2]))
training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,2,2]))
training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,2,2]))
training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,2,2]))
training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,2,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))




# 节点的特征
## 只看前两维，第一个<=120, 第二个<=3

# x = torch.tensor([[80,2,5,3,4,2,3,2,4], [12,1,3,5,3,0,1,2,2], [56,0,2,5,2,0,1,1,5]], dtype=torch.long)

x = torch.tensor([[20,0,1,2,4,5,2,3,1], [60,1,4,5,3,0,2,0,0], [100,2,4,5,3,0,2,0,0],[100,2,4,5,3,0,2,0,0]], dtype=torch.long)
edge_index = torch.tensor([[0, 1, 1, 2,3],
                           [1, 0, 2, 1,2]], dtype=torch.long)
edge_attr = torch.tensor([[0,0,1], [0,0,1], [0,0,1], [0,0,1],[0,0,1]], dtype=torch.long)
smiles = ''
text = "Two."


training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text,cand = [2,0,1]))
training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text,cand = [2,0,1]))
training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text,cand = [2,0,1]))
training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text,cand = [2,0,1]))
training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text,cand = [2,0,1]))
training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text,cand = [2,0,1]))
training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text,cand = [2,0,1]))
training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text,cand = [2,0,1]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))

aaa = myown()

aaa.process_save(training_data)



# training_data = []
#
# data_1 = dict()
#
# data_1["x"] = [[1,2],[2,3],[3,1],[2,2]]
# data_1["edge_index"] = [[2,1,2],[1,2,3]]
# data_1["edge_attr"] = [[0],[0],[0]]
# data_1["text"] = "This is a molecular."
# data_1["cid"] = '1'
# data_1["iupac"] = " "
# data_1["smiles"] = " "
#
# training_data.append(data_1)
#
# data_2 = dict()
#
# data_2["x"] = [[1,2],[2,3],[3,1],[2,2]]
# data_2["edge_index"] = [[2,1,2],[1,2,3]]
# data_2["edge_attr"] = [[0],[0],[0]]
# data_2["text"] = "This is a molecular."
# data_2["cid"] = '2'
# data_2["iupac"] = " "
# data_2["smiles"] = " "
#
# training_data.append(data_2)




# torch.save(data_list,"./pretrain.pt")
# torch.save(data_list,"./train.pt")
# torch.save(data_list,"./valid.pt")
# torch.save(data_list,"./test.pt")
