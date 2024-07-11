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


# flow_num = 30
# node_num = 20
# link_num = 80

# flow_num = 6
# node_num = 7
# link_num = 20

## opt: 87500
flow_num = 16
node_num = 7
link_num = 20

# SEED_A = 3
SEED_A = 2

random.seed(SEED_A)

dmd_max = 2000

# cap_max = 3000
# mini_unit = 100
mini_unit = 500
max_fiber = 3
capacity_per_fiber = 1000

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
    # ip_cost.append(random.randint(1,10))
    ip_cost.append(1)



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
    # dmd_list.append(random.randint(100, 2000))
    dmd_list.append(1000)

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
# dmd_max = 6000
#
# cap_max = 3000
# # mini_unit = 100
# mini_unit = 500
# max_fiber = 3
# capacity_per_fiber = 1000

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





# my_flow = {"src":[2,4,3,1,18,17,3,2,16,17,18,16], "dst":[3,0,1,2,3,4,2,18,17,18,3,3], "dmd": [200,300,400,2000,200,1000.5,1200.5,1800,400,1500,2000,100]}
# # 双向的link
# my_link = {"from":[0,0,0,0,1,1,1,1,2,2,2,3,3,3,3,4,4,4,4,2,18,17,3,17,18,16,18,16], "to":[1,2,3,4,0,2,3,4,0,1,4,0,1,2,4,0,1,2,3,3,2,0,18,18,17,17,16,2], "cap": [2000 for i in range(28)]}
# my_node = list(range(20))
# dmd_max = 6000
# cap_max = 2000

# out_dict = {}
#
# for i in range(len(my_node)):
#     out_dict[str(i)] = []
#     out_link_idx = []
#     for link_idx in range(len(my_link["from"])):
#         if my_link["from"][link_idx] == i:
#             out_link_idx.append(link_idx)
#     out_dict[str(i)] = out_link_idx
#
# in_dict = {}
#
# for i in range(len(my_node)):
#     in_dict[str(i)] = []
#     in_link_idx = []
#     for link_idx in range(len(my_link["to"])):
#         if my_link["to"][link_idx] == i:
#             in_link_idx.append(link_idx)
#     in_dict[str(i)] = in_link_idx






# print(my_flow["dst"])



# dmd_max:flow最大的demand情况

def my_solver_LP(my_flow, my_link, my_node,link_num,flow_num,dmd_max,allocated_capacity):
    # flow_cnt = len(my_flow["src"])
    # link_cnt = len(my_link["from"])
    # node_cnt = len(my_node)

    initial_LP = gp.Model('initial LP')

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

    x = [[0 for j in range(flow_num)] for i in range(link_num)]

    # for i in range(link_cnt*flow_cnt):  # 创建两个决策变量
    #     # 下界lb为0，上界ub为正无穷，变量类型vtype为连续型，变量名称name为x0和x1
    #     x[i] = initial_LP.addVar(lb=0, ub=dmd_max, vtype=GRB.CONTINUOUS, name='x_' + str(i))

    for i in range(link_num):
        for j in range(flow_num):
            # 下界lb为0，上界ub为正无穷，变量类型vtype为连续型，变量名称name为x0和x1
            x[i][j] = initial_LP.addVar(lb=0, ub=dmd_max, vtype=gp.GRB.CONTINUOUS, name=str(i)+"th link and "+str(j)+"th flow")
            # x[i][j] = initial_LP.addVar(lb=0, ub=dmd_max, vtype=GRB.INTEGER,
            #                             name=str(i) + "th link and " + str(j) + "th flow")


    # initial_LP.setObjective(100 * x[0] + 150 * x[1], GRB.MINIMIZE)  # 目标函数，设置为最大化MAXIMIZE
    # initial_LP.setObjective(
    #     gp.quicksum(ip_cost[i] * cap[i] * mini_unit  for i in range(link_num)), gp.GRB.MINIMIZE)
    initial_LP.setObjective(np.sum(x), gp.GRB.MINIMIZE)  # 目标函数，设置为最大化MAXIMIZE
    # initial_LP.setObjective(gp.quicksum(ip_cost[i] * x[i] for i in range(link_num)), gp.GRB.MINIMIZE)





    # initial_LP.addConstr(2 * x[0] + x[1] <= 10)  # 约束条件1
    # initial_LP.addConstr(3 * x[0] + 6 * x[1] <= 40)  # 约束条件2
    # initial_LP.addConstr(2 * x[0][0] + x[1][0] >= 10)
    for i in range(link_num):
        # for j in range(flow_cnt):
        #     initial_LP.addConstr(x[i][j] <= my_link["cap"][i])
        # initial_LP.addConstr(np.sum(x[i]) <= my_link["cap"][i])
        initial_LP.addConstr(np.sum(x[i]) <= allocated_capacity[i])

    for j in range(flow_num):
        curr_src = my_flow["src"][j]
        curr_dst = my_flow["dst"][j]
        curr_dmd = my_flow["dmd"][j]

        a = 0
        b = 0
        if len(out_dict[str(curr_src)]) > 0:
            for s in out_dict[str(curr_src)]:
                a += x[s][j]
        if len(in_dict[str(curr_src)]) > 0:
            for s in in_dict[str(curr_src)]:
                b += x[s][j]


        initial_LP.addConstr(a-b==my_flow["dmd"][j])

        a = 0
        b = 0
        if len(out_dict[str(curr_dst)]) > 0:
            for s in out_dict[str(curr_dst)]:
                a += x[s][j]
        if len(in_dict[str(curr_dst)]) > 0:
            for s in in_dict[str(curr_dst)]:
                b += x[s][j]

        initial_LP.addConstr(b - a == my_flow["dmd"][j])

        for other_node in my_node:
            if other_node is not curr_src and other_node is not curr_dst:
                a = 0
                b = 0
                if len(out_dict[str(other_node)]) > 0:
                    for s in out_dict[str(other_node)]:
                        a += x[s][j]
                if len(in_dict[str(other_node)]) > 0:
                    for s in in_dict[str(other_node)]:
                        b += x[s][j]

                initial_LP.addConstr(a - b == 0)
    # try:
    #     initial_LP.optimize()  # 调用求解器
    #     return True
    # except:
    #     return False

    initial_LP.optimize()

    try:
        print(initial_LP.ObjVal)
        return True
    except:
        return False



    # for var in initial_LP.getVars():
    #     print(var.Varname, '=', var.x)

import time
s1 = time.time()

print(my_solver_LP(my_flow, my_link, my_node,link_num,flow_num,dmd_max=dmd_max,allocated_capacity=[max_fiber*capacity_per_fiber for i in range(link_num)]))

s2 = time.time()

print(str(s2-s1)+ " s")