import torch
from torch.utils.data import DataLoader

from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import InMemoryDataset
from ILP import my_solver_ILP
import numpy as np
import random
import datetime


def my_level(raw, min_val, max_val, num):
    final = 0
    gran = (max_val-min_val)/num

    final = (raw // gran) - (min_val // gran)
    if final > num-1:
        final = num -1
    elif final < 0:
        final = 0
    return final

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


import time
import networkx as nx
import math

# from LM import my_solver_LP



## opt: 344278

## demand 降序：364512

## random 5: 367230
## random 4: 361741
## random 3: 363657
## random 2: 361000
## random 1: 365870








## opt: 333554
## demand 降序：361323
## 1000次random最佳：343148
## MOOD 3: 355793
## random 5: 367659
## random 4: 363167
## random 3: 366210
## random 2: 366398
## random 1: 362178

## 只看IP cost
test_mood = False
v2_mood = True
is_training = True
window_size = 2
per_network_sample = 1500
# per_network_sample = 16
level_a = 7
level_b = 30
level_c = 10

## 1: RANDOM  2: DESCENDING  3: SP上的占有率
MOOD = 3

# initial_LP = gp.Model()

fiber_cost = []
ip_cost = []


# flow_num = 100
# node_num = 30
# link_num = 200





# ## 229000
# flow_num = 20
# node_num = 20
# link_num = 80

## 355000
flow_num = 30
node_num = 20
link_num = 80



dmd_max = 2000

# cap_max = 3000
# mini_unit = 100
mini_unit = 500
max_fiber = 3
capacity_per_fiber = 1000


SEED_A = 3







# fiber_cost = 100
random.seed(SEED_A)

from_list = []
to_list = []
overall_list = []


i = 0
while i < link_num:
    my_from = random.randint(0, node_num-1)
    my_to = random.randint(0, node_num-1)
    if my_to == my_from:
        my_to = (my_to + 1) % node_num
    if [my_from, my_to] not in overall_list:
        overall_list.append([my_from,my_to])
        from_list.append(my_from)
        to_list.append(my_to)
        i += 1
# exit(0)

# for i in range(node_num):
#     for _ in range(node_num-1):
#         from_list.append(i)
#
# for i in range(node_num):
#     for j in range(node_num):
#         if not (j == i):
#             to_list.append(j)




## fiber cost 在那个范围之间分布
for i in range(len(from_list)):
    # random.seed(time.time())
    # fiber_cost.append(random.randint(100,200))
    fiber_cost.append(0)
    ip_cost.append(random.randint(1,10))



src_list = []
dst_list = []
dmd_list = []



for i in range(flow_num):
    # random.seed(time.time())

    src_list.append(random.randint(0, node_num-1))
    # random.seed(time.time())

    aa = random.randint(0, node_num - 1)

    if aa == src_list[i]:
        aa = (aa + 1)%node_num
    dst_list.append(aa)
    # random.seed(time.time())
    ## dmd 在哪些地方 浮动
    dmd_list.append(random.randint(100, 2000))

    # print(src_list[i])
    # print(dst_list[i])

# exit(0)



# my_flow = {"src":[4,3,1,8,7,3,2,4,9,6,5,4,9,11, 0,1,2,3,4,2,1,2,4,3,3,1,8,3], "dst":[0,1,2,3,4,2,1,2,4,3,3,1,8,3,4,3,1,8,7,3,2,4,9,6,5,4,9,11],
#            "dmd": [300,400,2000,200,1000.5,1200.5,1800,400,1500,2000,100,600,200,1000, 300,400,2000,200,1000.5,1200.5,1800,400,1500,2000,100,600,200,1000]}
my_flow = {"src":src_list, "dst":dst_list,
           "dmd": dmd_list}
# 双向的link
# my_link = {"from":from_list, "to":to_list, "cap": [2000 for i in range(node_num*(node_num-1))]}
my_link = {"from":from_list, "to":to_list}
my_node = list(range(node_num))

## abandoned


out_dict = {}

for i in range(len(my_node)):
    out_dict[str(i)] = []
    out_link_idx = []
    for link_idx in range(len(my_link["from"])):
        if my_link["from"][link_idx] == i:
            out_link_idx.append(link_idx)
    out_dict[str(i)] = out_link_idx

in_dict = {}

for i in range(len(my_node)):
    in_dict[str(i)] = []
    in_link_idx = []
    for link_idx in range(len(my_link["to"])):
        if my_link["to"][link_idx] == i:
            in_link_idx.append(link_idx)
    in_dict[str(i)] = in_link_idx



cost_list = []



Gbase = nx.DiGraph()
Gbase.add_nodes_from(my_node)
edge_list = []
for i in range(len(from_list)):
    edge_list.append((from_list[i],to_list[i]))
Gbase.add_edges_from(edge_list)

flow_seq = list(range(flow_num))

raw_x_list = [[0 for _ in range(3)] for _ in range(link_num)]
smiles = ''

## three features: 已经占了多少比例（动态变化）；预计flow占多少比例（可以超过100%）；normalized ip cost

## third feature
for j in range(link_num):
    raw_x_list[j][2] = ip_cost[j]

# second feature
for i in range(flow_num):
    curr_flow = i
    path = nx.dijkstra_path(Gbase, source=my_flow["src"][curr_flow], target=my_flow["dst"][curr_flow])
    for node in range(len(path) - 1):
        a = path[node]
        b = path[node + 1]
        for s in range(len(from_list)):
            if a == from_list[s] and b == to_list[s]:
                raw_x_list[s][1] += my_flow["dmd"][curr_flow]

## transformed edges
edge_index_from = []
edge_index_to = []

for s in range(link_num):
    target = my_link["to"][s]
    for cand in range(link_num):
        if (not (s == cand)) and target == my_link["from"][cand]:
            edge_index_from.append(s)
            edge_index_to.append(cand)

edge_index = [edge_index_from, edge_index_to]


if is_training:

    b_list = []
    for i in range(link_num):
        b_list.append(raw_x_list[i][1])
    b_max = np.max(b_list)




    ## first feature: 需要尝试不同分配情况
    opt_cap = my_solver_ILP(my_flow, my_link, my_node,dmd_max,max_fiber,capacity_per_fiber,mini_unit)
    for i in range(len(opt_cap)):
        opt_cap[i] = opt_cap[i] * mini_unit

    training_data = []

    for sam in range(per_network_sample):

        cand_list = [i for i in range(len(opt_cap))]
        ## false sample, 也就是不需要加的
        false_list = random.sample(cand_list, window_size-1)
        ## true sample, 也就是需要加的
        true_cand_list = []
        for i in range(len(cand_list)):
            if opt_cap[i] > 0:
                true_cand_list.append(i)
        true_list = random.sample(true_cand_list, 1)


        for i in range(len(cand_list)):
            if i in false_list:
                raw_x_list[i][0] = opt_cap[i]
            elif i in true_list:
                random.seed(datetime.datetime.now())
                raw_x_list[i][0] = random.randint(0,opt_cap[i]/mini_unit-1) * mini_unit
            else:
                random.seed(datetime.datetime.now())
                # print(raw_x_list)
                # print(i)
                raw_x_list[i][0] = random.randint(0, opt_cap[i] / mini_unit) * mini_unit

        random.seed(datetime.datetime.now())

        true_pos = random.randint(0,window_size-1)

        # print(true_pos)

        cand = false_list
        cand.insert(true_pos,true_list[0])

        if v2_mood:
            cand_list = [i for i in range(len(opt_cap))]
            cand = random.sample(cand_list, window_size)
            diff_list = []
            for m in range(link_num):
                raw_x_list[m][0] = random.randint(0, opt_cap[i] / mini_unit) * mini_unit
            for j in range(window_size):
                i = cand[j]
                diff_list.append(opt_cap[i] - raw_x_list[i][0])
            true_pos = np.argmax(diff_list)
            print(true_pos)



        if test_mood:
            cand_cand = [i for i in range(link_num)]
            cand = random.sample(cand_cand, window_size)
            cand_ip_list = []
            for i in range(len(cand)):
                cand_ip_list.append(ip_cost[cand[i]])
            true_pos = np.argmin(cand_ip_list)
            for i in range(link_num):
                raw_x_list[i][0] = 0




        text = ""
        if true_pos == 0:
            text = "One."
        elif true_pos == 1:
            text = "Two."
        elif true_pos == 2:
            text = "Three."
        elif true_pos == 3:
            text = "Four."
        elif true_pos == 4:
            text = "Five."
        elif true_pos == 5:
            text = "Six."
        elif true_pos == 6:
            text = "Seven."
        elif true_pos == 7:
            text = "Eight."
        elif true_pos == 8:
            text = "Nine."
        elif true_pos == 9:
            text = "Ten."





        for i in range(len(opt_cap)):
            aa = my_level(raw_x_list[i][0], 0, max_fiber*capacity_per_fiber, level_a)
            raw_x_list[i][0] = aa

            if sam == 0:
                bb = my_level(raw_x_list[i][1], 0, b_max, level_b)
                raw_x_list[i][1] = bb

                cc = my_level(raw_x_list[i][2], np.min(ip_cost), np.max(ip_cost), level_c)
                raw_x_list[i][2] = cc


        x = torch.tensor(raw_x_list, dtype=torch.long)




        edge_index = torch.tensor(edge_index,dtype=torch.long)

        edge_attr = torch.tensor([[0, 0, 1] for _ in range(len(edge_index_from))], dtype=torch.long)

        training_data.append(
            Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles, text=text, cand=cand))



    aaa = myown()

    aaa.process_save(training_data)

else:
    b_list = []
    for i in range(link_num):
        b_list.append(raw_x_list[i][1])
    b_max = np.max(b_list)

    training_data = []

    for sam in range(per_network_sample):

        cand_list = [i for i in range(link_num)]
        ## false sample, 也就是不需要加的
        false_list = random.sample(cand_list, window_size - 1)
        ## true sample, 也就是需要加的
        # true_cand_list = []
        # for i in range(len(cand_list)):
        #     if opt_cap[i] > 0:
        #         true_cand_list.append(i)
        true_list = random.sample(cand_list, 1)

        for i in range(len(cand_list)):
            if i in false_list:
                raw_x_list[i][0] = 0
            elif i in true_list:
                random.seed(datetime.datetime.now())
                raw_x_list[i][0] = 0
            else:
                random.seed(datetime.datetime.now())
                # print(raw_x_list)
                # print(i)
                raw_x_list[i][0] = 0

        random.seed(datetime.datetime.now())

        true_pos = random.randint(0, window_size - 1)

        # print(true_pos)

        cand = false_list
        cand.insert(true_pos, true_list[0])

        text = "One"


        for i in range(link_num):
            aa = my_level(raw_x_list[i][0], 0, max_fiber * capacity_per_fiber, level_a)
            raw_x_list[i][0] = aa

            if sam==0:
                bb = my_level(raw_x_list[i][1], 0, b_max, level_b)
                raw_x_list[i][1] = bb

                cc = my_level(raw_x_list[i][2], np.min(ip_cost), np.max(ip_cost), level_c)
                raw_x_list[i][2] = cc

        x = torch.tensor(raw_x_list, dtype=torch.long)

        # print(x)

        edge_index = torch.tensor(edge_index, dtype=torch.long)

        edge_attr = torch.tensor([[0, 0, 1] for _ in range(len(edge_index_from))], dtype=torch.long)

        training_data.append(
            Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles, text=text, cand=cand))

    aaa = myown()

    aaa.process_save(training_data)



# training_data = []
# # 由于是无向图，因此有 4 条边：(0 -> 1), (1 -> 0), (1 -> 2), (2 -> 1)
#
# # 节点的特征
#
# # x = torch.tensor([[5,0,4,5,3,0,2,0,0], [5,0,4,5,3,0,2,0,0], [5,0,4,5,3,0,2,0,0]], dtype=torch.long)
# x = torch.tensor([[20,0,1,2,4,5,2,3,1], [60,1,4,5,3,0,2,0,0], [100,2,4,5,3,0,2,0,0]], dtype=torch.long)
# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# edge_attr = torch.tensor([[0,0,1], [0,0,1], [0,0,1], [0,0,1]], dtype=torch.long)
# smiles = ''
# text = "One."
# # batch = torch.tensor([0,0,0,1,1,1], dtype=torch.float)
#
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,2,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,2,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,2,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,2,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,2,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,2,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,2,2]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,2,2]))
# # training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# # training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# # training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# # training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# # training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# # training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# # training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# # training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
#
#
#
#
# # 节点的特征
# ## 只看前两维，第一个<=120, 第二个<=3
#
# # x = torch.tensor([[80,2,5,3,4,2,3,2,4], [12,1,3,5,3,0,1,2,2], [56,0,2,5,2,0,1,1,5]], dtype=torch.long)
#
# x = torch.tensor([[20,0,1,2,4,5,2,3,1], [60,1,4,5,3,0,2,0,0], [100,2,4,5,3,0,2,0,0],[100,2,4,5,3,0,2,0,0]], dtype=torch.long)
# edge_index = torch.tensor([[0, 1, 1, 2,3],
#                            [1, 0, 2, 1,2]], dtype=torch.long)
# edge_attr = torch.tensor([[0,0,1], [0,0,1], [0,0,1], [0,0,1],[0,0,1]], dtype=torch.long)
# smiles = ''
# text = "Two."
#
#
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text,cand = [2,0,1]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text,cand = [2,0,1]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text,cand = [2,0,1]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text,cand = [2,0,1]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text,cand = [2,0,1]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text,cand = [2,0,1]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text,cand = [2,0,1]))
# training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text,cand = [2,0,1]))
# # training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# # training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# # training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# # training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# # training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# # training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# # training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))
# # training_data.append(Data(x=x, edge_index=edge_index,edge_attr = edge_attr, smiles = smiles, text=text, cand = [0,1,2]))





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
