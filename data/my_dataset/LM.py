import gurobipy as gp
import numpy as np
import random
import networkx as nx

'''定义了一个线性松弛问题，并用Gurobi求解'''
# initial_LP = gp.Model('initial LP')  # 定义变量initial_LP，调用Gurobi的Model，选择Initial Programming（整数规划）模型
# x = {}  # 创建一个空字典来存储决策变量

fiber_cost = []
ip_cost = []

# flow_num = 100
# node_num = 30
# link_num = 300


## > 1 day unsplittable
# flow_num = 60
# node_num = 20
# link_num = 150

## > 1 day splittable
# flow_num = 60
# node_num = 20
# link_num = 150


# flow_num = 50
# node_num = 20
# link_num = 100


# ## 139500
# flow_num = 20
# node_num = 10
# link_num = 50

## 139500
# flow_num = 40
# node_num = 30
# link_num = 200






# # opt:197000
# flow_num = 20
# node_num = 20
# link_num = 80

# # opt:298500
# flow_num = 30
# node_num = 20
# link_num = 80

# ## opt: 99500
# flow_num = 6
# node_num = 7
# link_num = 20


## opt: 87500
flow_num = 13
node_num = 7
## is even
link_num = 30


# ## opt: 87500
# flow_num = 30
# node_num = 15
# link_num = 100


dmd_max = 2000

# cap_max = 3000
# mini_unit = 100
mini_unit = 500
max_fiber = 3
capacity_per_fiber = 1000


# SEED_A = 3
SEED_A = 4






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
    if ([my_from, my_to] not in overall_list) and ([my_to, my_from] not in overall_list):
        overall_list.append([my_from,my_to])
        overall_list.append([my_to, my_from])
        from_list.append(my_from)
        to_list.append(my_to)
        from_list.append(my_to)
        to_list.append(my_from)
        i += 2
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
for i in range(link_num//2):
    # random.seed(time.time())
    # fiber_cost.append(random.randint(100,200))
    # fiber_cost.append(0)
    ip_cost.append(random.randint(1,10))
    ## 双向ip cost相等
    # ip_cost.append(1)




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
    # dmd_list.append(1000)

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

## new
my_path = []
Gbase = nx.DiGraph()
Gbase.add_nodes_from(my_node)
edge_list = []
for i in range(len(from_list)):
    # edge_list.append((from_list[i],to_list[i]))
    Gbase.add_weighted_edges_from([(from_list[i],to_list[i],{'weight':ip_cost[i//2]})])

def get_k_shortest_paths(graph, source, target, k):
    paths = nx.shortest_simple_paths(graph, source, target)
    shortest_paths = []
    for path in paths:
        shortest_paths.append(path)
        if len(shortest_paths) == k:
            break
    return shortest_paths


# my_path: [[[a,b],[b,c],]
for i in range(flow_num):
    my_path.append([])
    curr_flow = i
    k_shortest_paths = get_k_shortest_paths(Gbase, source=my_flow["src"][curr_flow], target=my_flow["dst"][curr_flow],k=4)
    for i, path in enumerate(k_shortest_paths):
        my_path[-1].append(path)
#
#     print(my_path[-1])
# exit(0)



# print(my_flow["dst"])


# dmd_max:flow最大的demand情况

def my_solver_LP(ip_cost,my_flow, my_link, my_node,link_num,flow_num,dmd_max,my_path,allocated_capacity):
    # flow_cnt = len(my_flow["src"])
    # link_cnt = len(my_link["from"])
    # node_cnt = len(my_node)

    initial_LP = gp.Model('initial LP')

    flow_cnt = len(my_flow["src"])
    link_cnt = len(my_link["from"])
    node_cnt = len(my_node)

    # x = [[0 for j in range(flow_cnt)] for i in range(link_cnt)]
    # x[i][j]: allocated data rate of i^th flow's j^th path
    x = [[0 for j in range(4)] for i in range(flow_cnt)]
    f = [[0 for j in range(link_cnt)] for i in range(flow_cnt)]

    # for i in range(link_cnt*flow_cnt):  # 创建两个决策变量
    #     # 下界lb为0，上界ub为正无穷，变量类型vtype为连续型，变量名称name为x0和x1
    #     x[i] = initial_LP.addVar(lb=0, ub=dmd_max, vtype=GRB.CONTINUOUS, name='x_' + str(i))


    for i in range(flow_cnt):
        # 下界lb为0，上界ub为正无穷，变量类型vtype为连续型，变量名称name为x0和x1
        for j in range(link_cnt):
            f[i][j] = initial_LP.addVar(lb=0, ub=dmd_max, vtype=gp.GRB.CONTINUOUS,
                                        name=str(i) + "th flow and " + str(j) + "th link")

        for j in range(4):
            x[i][j] = initial_LP.addVar(lb=0, ub=dmd_max, vtype=gp.GRB.CONTINUOUS,
                                        name=str(i) + "th flow and " + str(j) + "th split")

    for i in range(flow_cnt):
        for j in range(4):
            if j >= len(my_path[i]):
                initial_LP.addConstr(x[i][j] == 0)

        initial_LP.addConstr(np.sum(x[i]) == my_flow["dmd"][i])

    initial_LP.setObjective(
        gp.quicksum(ip_cost[i] for i in range(link_cnt // 2)), gp.GRB.MINIMIZE)

    ## calculate f[i][j]: the allocated data rate for i^th flow on j^th link
    for i in range(flow_cnt):
        # by link index
        split1_link_list = []
        split2_link_list = []
        split3_link_list = []
        split4_link_list = []
        for m in range(link_cnt):
            link_set = []
            this_from = my_link["from"][m]
            this_to = my_link["to"][m]
            split_cnt = 0
            for split in range(len(my_path[i])):
                split_cnt += 1
                for imm in range(len(my_path[i][split]) - 1):
                    if this_from == my_path[i][split][imm] and this_to == my_path[i][split][imm + 1]:
                        if split_cnt == 1:
                            split1_link_list.append(m)
                            link_set.append(0)
                        elif split_cnt == 2:
                            split2_link_list.append(m)
                            link_set.append(1)
                        elif split_cnt == 3:
                            split3_link_list.append(m)
                            link_set.append(2)
                        elif split_cnt == 4:
                            split4_link_list.append(m)
                            link_set.append(3)
            initial_LP.addConstr(gp.quicksum(x[i][t] for t in link_set) == f[i][m])

    for i in range(link_cnt // 2):
        # for j in range(flow_cnt):
        # initial_LP.addConstr(x[i][j] <= cap[i])

        initial_LP.addConstr(gp.quicksum(f[s][2 * i] + f[s][2 * i + 1] for s in range(flow_cnt)) <= allocated_capacity[i])
        # initial_LP.addConstr(cap[i] * mini_unit  <= U[i] * capacity_per_fiber)

    # for i in range(link_cnt // 2):
    #     # if link_cnt % 2 == 0:
    #     #     initial_LP.addConstr(cap[i] * mini_unit + cap[(i + 1) % link_cnt] * mini_unit <= U[i] * capacity_per_fiber)
    #     # initial_LP.addConstr(cap[i] * mini_unit  <= U[i] * capacity_per_fiber)
    #     initial_LP.addConstr(cap[i] * mini_unit <= max_fiber * capacity_per_fiber)

    initial_LP.optimize()  # 调用求解器

    try:
        print(initial_LP.ObjVal)
        return True
    except:
        return False



    # for var in initial_LP.getVars():
    #     print(var.Varname, '=', var.x)

import time
s1 = time.time()

print(my_solver_LP(ip_cost,my_flow, my_link, my_node,link_num,flow_num,dmd_max=dmd_max,my_path=my_path,allocated_capacity=[0 for i in range(link_num//2)]))

s2 = time.time()

print(str(s2-s1)+ " s")
