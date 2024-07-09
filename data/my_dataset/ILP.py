import gurobipy as gp
import numpy as np
import random
import time

'''定义了一个线性松弛问题，并用Gurobi求解'''
# initial_LP = Model('initial ILP')  # 定义变量initial_LP，调用Gurobi的Model，选择Initial Programming（整数规划）模型
# x = {}  # 创建一个空字典来存储决策变量



## 12 nodes,  130 全连接 links， 40 个 flow；demand_max=2000, minimal unit = 100， cap_max = 5000    1.03 s
## 12 nodes,  130 全连接 links， 60 个 flow；demand_max=2000, minimal unit = 100  ， cap_max = 5000   177 s
## 12 nodes,  130 全连接 links， 100 个 flow；demand_max=2000, minimal unit = 100 ， cap_max = 5000    29.61 s
## 12 nodes,  130 全连接 links， 150 个 flow；demand_max=2000, minimal unit = 100 ， cap_max = 15000    476 s
## 15 nodes,  210 全连接 links， 150 个 flow；demand_max=2000, minimal unit = 100 ， cap_max = 15000    158 s
## 15 nodes,  210 全连接 links， 150 个 flow；demand_max=2000, minimal unit = 100 ， cap_max = 10000    1019 s
## 12 nodes,  130 links,       100 个 flow，demand = [100, 2000],minimal unit = 100, fiber cost = [100, 200], cap_max = 12000      3475 s !!!



# initial_LP = gp.Model()

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

# opt:298500
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






# print(my_flow["dst"])

def my_solver_ILP(my_flow, my_link, my_node,dmd_max,max_fiber,capacity_per_fiber,mini_unit):
    initial_LP = gp.Model()
    flow_cnt = len(my_flow["src"])
    link_cnt = len(my_link["from"])
    node_cnt = len(my_node)

    x = [[0 for j in range(flow_cnt)] for i in range(link_cnt)]
    f = [[0 for j in range(flow_cnt)] for i in range(link_cnt)]
    cap = [0 for i in range(link_cnt)]
    U = [0 for i in range(link_cnt)]
    final = [0 for i in range(link_cnt)]

    # for i in range(link_cnt*flow_cnt):  # 创建两个决策变量
    #     # 下界lb为0，上界ub为正无穷，变量类型vtype为连续型，变量名称name为x0和x1
    #     x[i] = initial_LP.addVar(lb=0, ub=dmd_max, vtype=GRB.CONTINUOUS, name='x_' + str(i))

    for i in range(link_cnt):
        # cap[i] = initial_LP.addVar(lb=0, ub=cap_max/mini_unit, vtype=gp.GRB.INTEGER,
        #                             name=str(i) + "th link capacity")
        cap[i] = initial_LP.addVar(lb=0, ub=capacity_per_fiber*max_fiber/mini_unit, vtype=gp.GRB.INTEGER,
                                   name=str(i) + "th link capacity")
        # U[i] = initial_LP.addVar(lb=0, ub=1, vtype=gp.GRB.INTEGER,
        #                            name=str(i) + "th link capacity's boolean")
        # U[i] = initial_LP.addVar(lb=0 , ub = max_fiber, vtype=gp.GRB.INTEGER,
        #                          name=str(i) + "th link capacity's boolean")
        # cap[i] = initial_LP.addVar( vtype=gp.GRB.INTEGER,
        #                            name=str(i) + "th link capacity")
        # if cap[i] >= 2:
        #     final[i] = cap[i] + 100
        # else:
        #     final[i] = 0

        # final[i] = np.max(3,cap[i])

        for j in range(flow_cnt):
            # 下界lb为0，上界ub为正无穷，变量类型vtype为连续型，变量名称name为x0和x1
            x[i][j] = initial_LP.addVar(lb=0, ub=dmd_max, vtype=gp.GRB.CONTINUOUS, name=str(i)+"th link and "+str(j)+"th flow")
            # x[i][j] = initial_LP.addVar(lb=0, ub=1, vtype=gp.GRB.BINARY,
            #
            #                             name=str(i) + "th link and " + str(j) + "th flow")


            # f[i][j] = x[i][j] * my_flow["dmd"][j]


            # x[i][j] = initial_LP.addVar(lb=0, ub=dmd_max, vtype=GRB.INTEGER,
            #                             name=str(i) + "th link and " + str(j) + "th flow")


    # initial_LP.setObjective(100 * x[0] + 150 * x[1], GRB.MINIMIZE)  # 目标函数，设置为最大化MAXIMIZE
    # initial_LP.setObjective(gp.quicksum(ip_cost[i]*cap[i]* mini_unit+U[i]*fiber_cost[i] for i in range(link_cnt)), gp.GRB.MINIMIZE)  # 目标函数，设置为最大化MAXIMIZE
    initial_LP.setObjective(
        gp.quicksum(ip_cost[i] * cap[i] * mini_unit for i in range(link_cnt)), gp.GRB.MINIMIZE)

    # initial_LP.addConstr(2 * x[0] + x[1] <= 10)  # 约束条件1
    # initial_LP.addConstr(3 * x[0] + 6 * x[1] <= 40)  # 约束条件2
    # initial_LP.addConstr(2 * x[0][0] + x[1][0] >= 10)
    for i in range(link_cnt):
        # for j in range(flow_cnt):
            # initial_LP.addConstr(x[i][j] <= cap[i])
        initial_LP.addConstr(np.sum(x[i]) <= cap[i]* mini_unit)
        # initial_LP.addConstr(cap[i] * mini_unit  <= U[i] * capacity_per_fiber)

    for i in range(link_cnt):
        # if link_cnt % 2 == 0:
        #     initial_LP.addConstr(cap[i] * mini_unit + cap[(i + 1) % link_cnt] * mini_unit <= U[i] * capacity_per_fiber)
        # initial_LP.addConstr(cap[i] * mini_unit  <= U[i] * capacity_per_fiber)
        initial_LP.addConstr(cap[i] * mini_unit <= max_fiber * capacity_per_fiber)

    for j in range(flow_cnt):
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

    initial_LP.optimize()  # 调用求解器

    # for var in initial_LP.getVars():
    #     print(var.Varname, '=', var.x)
    print(initial_LP.ObjVal)

    capacity_list = []
    for var in initial_LP.getVars():
        if var.Varname[-1] == "y":
            capacity_list.append(var.x)

    return capacity_list


import time

# print(out_dict["18"])
#
s1 = time.time()
my_solver_ILP(my_flow, my_link, my_node,dmd_max,max_fiber,capacity_per_fiber,mini_unit)
s2 = time.time()

print(str(s2-s1)+ " s")

