# import gurobipy as gp
import numpy as np
import random
import time
import networkx as nx
import math

from LM import my_solver_LP
import networkx as nx


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




## 1: RANDOM  2: DESCENDING  3: SP上的占有率
MOOD = 2

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

# 355000
# flow_num = 30
# node_num = 20
# link_num = 80

# flow_num = 6
# node_num = 7
# link_num = 20

flow_num = 14
node_num = 8
link_num = 20

dmd_max = 2000

# cap_max = 3000
# mini_unit = 100
mini_unit = 500
max_fiber = 3
capacity_per_fiber = 1000


# SEED_A = 3
SEED_A = 2







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

## flow的顺序







Gbase = nx.DiGraph()
Gbase.add_nodes_from(my_node)
edge_list = []
for i in range(len(from_list)):
    edge_list.append((from_list[i],to_list[i]))
Gbase.add_edges_from(edge_list)

flow_seq = list(range(flow_num))

# flow scheduling seq
if MOOD == 1:
    random.seed(2)
    random.shuffle(flow_seq)


elif MOOD == 2:
    ## flow 按 demand 降序排列
    flow_seq_copy = []
    for i in range(len(flow_seq)):
        flow_seq_copy.append((i,dmd_list[i]))

    flow_seq_copy_2 = sorted(flow_seq_copy, key=lambda flow_seq_copy : flow_seq_copy[1], reverse=True)
    # flow_seq_copy_2 = sorted(flow_seq_copy, key=lambda flow_seq_copy : flow_seq_copy[1])

    # print(flow_seq_copy_2)
    # exit(0)

    flow_seq = []
    for i in range(len(flow_seq_copy_2)):
        flow_seq.append(flow_seq_copy_2[i][0])

elif MOOD == 3:
    remaining_capacity = []
    allocated_capacity = []

    for i in range(len(from_list)):
        remaining_capacity.append(capacity_per_fiber * max_fiber)
        allocated_capacity.append(0)
    my_copy_graph = nx.DiGraph()
    my_copy_graph.add_nodes_from(my_node)
    affordable_edge_list = []

    for i in range(len(from_list)):
        affordable_edge_list.append((from_list[i], to_list[i]))

    # my_copy_graph.add_edges_from(affordable_edge_list)

    for i in range(len(from_list)):
        if edge_list[i] in affordable_edge_list:
            # aa = mini_func((allocated_capacity[i] + my_flow["dmd"][flow_idx]))
            # bb = mini_func(allocated_capacity[i])
            # new_fiber = math.ceil(aa / capacity_per_fiber) - math.ceil(bb / capacity_per_fiber)
            my_copy_graph.add_edge(from_list[i], to_list[i],
                                   weight=capacity_per_fiber * max_fiber * ip_cost[i])
    edge_disjointness_dict = {}
    flow_score = []
    for m in range(len(affordable_edge_list)):
        edge_disjointness_dict[str(from_list[m])+" "+str(to_list[m])] = 0
    for i in range(flow_num):
        curr_flow = i
        path = nx.dijkstra_path(my_copy_graph, source=my_flow["src"][curr_flow], target=my_flow["dst"][curr_flow])
        for node in range(len(path) - 1):
            a = path[node]
            b = path[node + 1]
            edge_disjointness_dict[str(a) + " " + str(b)] += 1
    for i in range(flow_num):
        flow_score.append(0)
        curr_flow = i
        path = nx.dijkstra_path(my_copy_graph, source=my_flow["src"][curr_flow], target=my_flow["dst"][curr_flow])
        for node in range(len(path) - 1):
            a = path[node]
            b = path[node + 1]
            flow_score[i] += edge_disjointness_dict[str(a) + " " + str(b)]
    flow_seq_copy = []
    for i in range(len(flow_seq)):
        flow_seq_copy.append((i, flow_score[i]))

    flow_seq_copy_2 = sorted(flow_seq_copy, key=lambda flow_seq_copy: flow_seq_copy[1], reverse=False)
    # flow_seq_copy_2 = sorted(flow_seq_copy, key=lambda flow_seq_copy : flow_seq_copy[1])

    # print(flow_seq_copy_2)
    # exit(0)

    flow_seq = []
    for i in range(len(flow_seq_copy_2)):
        flow_seq.append(flow_seq_copy_2[i][0])










candidate_flow_seq = tuple(flow_seq)



remaining_capacity = []
allocated_capacity = []

for i in range(len(from_list)):
    remaining_capacity.append(capacity_per_fiber * max_fiber)
    allocated_capacity.append(0)




def mini_func(x):
    if x % mini_unit == 0:
        return x
    else:
        return x + (mini_unit - (x % mini_unit))

tot_cost = 0


if MOOD == 1 or MOOD == 2 or MOOD == 3:
    ## 先尝试通过unsplittable flow routing来开启capacity，可能会有flow无法routing
    allocated_flow_cnt = 0
    for t in range(len(flow_seq)):
        curr_flow = flow_seq[t]
        my_copy_graph = nx.DiGraph()
        my_copy_graph.add_nodes_from(my_node)

        affordable_edge_list = []

        for i in range(len(from_list)):
            if remaining_capacity[i] >= my_flow["dmd"][curr_flow]:
                affordable_edge_list.append((from_list[i], to_list[i]))

        # my_copy_graph.add_edges_from(affordable_edge_list)

        for i in range(len(from_list)):
            if edge_list[i] in affordable_edge_list:
                # aa = mini_func((allocated_capacity[i] + my_flow["dmd"][flow_idx]))
                # bb = mini_func(allocated_capacity[i])
                # new_fiber = math.ceil(aa / capacity_per_fiber) - math.ceil(bb / capacity_per_fiber)
                my_copy_graph.add_edge(from_list[i], to_list[i],
                                       weight=my_flow["dmd"][curr_flow] * ip_cost[i])

        try:
            path = nx.dijkstra_path(my_copy_graph, source=my_flow["src"][curr_flow], target=my_flow["dst"][curr_flow])
            allocated_flow_cnt += 1
            ## allocate the flow
            for node in range(len(path) - 1):
                a = path[node]
                b = path[node + 1]
                for s in range(len(from_list)):
                    if a == from_list[s] and b == to_list[s]:
                        incr_capacity = mini_func(allocated_capacity[s]+my_flow["dmd"][curr_flow]) - mini_func(allocated_capacity[s])
                        tot_cost += (ip_cost[s] * incr_capacity)
                        allocated_capacity[s] += incr_capacity
                        remaining_capacity[s] -= incr_capacity
                        break

        except:
            pass



    ## randomly add ip capacity until LP becomes feasible

    random_add_cnt = 0
    panbie = my_solver_LP(my_flow, my_link, my_node,link_num,flow_num,dmd_max,allocated_capacity)

    # print(my_solver_LP(my_flow, my_link, my_node,link_num,flow_num,dmd_max,[3000 for _ in range(link_num)]))
    #
    # exit(0)
    while not panbie:
        random_add_cnt += 1

        candidate_idx_list = []

        for i in range(len(remaining_capacity)):
            if remaining_capacity[i] > 0:
                candidate_idx_list.append(i)

        random.seed(2)

        s = random.choice(candidate_idx_list)

        allocated_capacity[s] += mini_unit
        remaining_capacity[s] -= mini_unit

        tot_cost += (mini_unit * ip_cost[s])

        panbie = my_solver_LP(my_flow, my_link, my_node,link_num,flow_num,dmd_max,allocated_capacity)

    print("allocated_flow_cnt")
    print(allocated_flow_cnt)

    print("random_add_cnt")
    print(random_add_cnt)


    print("tot_cost")
    print(tot_cost)

























