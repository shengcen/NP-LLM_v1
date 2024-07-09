"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import gurobipy as gp
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel

from lavis.models.blip2_models.blip2 import (
    # Blip2Base,
    disabled_train,
)
from model.blip2 import Blip2Base
from transformers import LlamaTokenizer
from model.modeling_llama import LlamaForCausalLM
import random
import datetime
import time
import networkx as nx
import math

test_mood = False
level_a = 7
level_b = 30
level_c = 10

fiber_cost = []
ip_cost = []

# flow_num = 100
# node_num = 30
# link_num = 300

window_size = 2

flow_num = 30
node_num = 20
link_num = 80

SEED_A = 3

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

def to_label(targets):
    answer = []
    for i in range(len(targets)):
        if targets[i] == "One.":
            answer.append(0)
        elif targets[i] == "Two.":
            answer.append(1)
        elif targets[i] == "Three.":
            answer.append(2)
        elif targets[i] == "Four.":
            answer.append(3)
        elif targets[i] == "Five.":
            answer.append(4)
        elif targets[i] == "Six.":
            answer.append(5)
        elif targets[i] == "Seven.":
            answer.append(6)
        elif targets[i] == "Eight.":
            answer.append(7)
        elif targets[i] == "Nine.":
            answer.append(8)
        elif targets[i] == "Ten.":
            answer.append(9)
    return answer


def to_answer(output_text, window_size):

    answer = 0
    if len(output_text) == 0 or 1:
        random.seed(datetime.datetime.now())
        answer = random.randint(0, window_size-1)
    else:
        my_word = output_text[0:2]
        print("my_word:")
        print(my_word)
        if my_word == 'On':
            answer = 0
        elif my_word == 'Tw':
            answer = 1
        elif my_word == 'Th':
            answer = 2
        elif my_word == 'Fo':
            answer = 3
        elif my_word == 'Fi':
            answer = 4
        elif true_pos == 'Si':
            answer = 5
        elif true_pos == 'Se':
            answer = 6
        elif true_pos == 'Ei':
            answer = 7
        elif true_pos == 'Ni':
            answer = 8
        elif true_pos == 'Te':
            answer = 9
        else:
            random.seed(datetime.datetime.now())
            answer = random.randint(0, window_size - 1)
    return answer



def my_level(raw, min_val, max_val, num):
    final = 0
    gran = (max_val-min_val)/num

    final = (raw // gran) - (min_val // gran)
    if final > num-1:
        final = num -1
    elif final < 0:
        final = 0
    return final


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

def transformed_edge_index(link_num, my_link):
    edge_index_from = []
    edge_index_to = []

    for s in range(link_num):
        target = my_link["to"][s]
        for cand in range(link_num):
            if (not (s == cand)) and target == my_link["from"][cand]:
                edge_index_from.append(s)
                edge_index_to.append(cand)
    edge_index = [edge_index_from, edge_index_to]
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index, len(edge_index_from)





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


 
llama_model_list = [
    "decapoda-research/llama-13b-hf",
    "decapoda-research/llama-7b-hf",
]

def mask_by_len(input, lens, fill_value=0):
    '''
    input: shape = [N, D]
    lens: shape = [N]
    '''
    mask = torch.arange(input.shape[1], device=input.device).reshape(1, -1)
    mask = mask < lens.reshape(-1, 1)
    input[mask] = fill_value
    return input

# @registry.register_model("blip2")
# @registry.register_model("blip2_feature_extractor")
class Blip2Llama(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """
    def __init__(
        self,
        bert_name,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        lora_tuning=False,
        peft_dir='',
        llm_model="decapoda-research/llama-7b-hf",
        prompt="",
        args=None,
    ):
        super().__init__()
        self.is_training = args.is_training
        self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio)
        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")
        
        self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq)
        ### remove the unused parameters
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        ## initialize opt model
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, padding_side='right')
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_model = LlamaForCausalLM.from_pretrained(llm_model, torch_dtype=torch.bfloat16)
        # self.llm_model = LlamaForCausalLM.from_pretrained(llm_model)
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        ## ADD
        self.llm_tokenizer.mol_token_id = self.llm_tokenizer("<mol>", add_special_tokens=False).input_ids[0]

        ## MY_ADD
        self.action_head = nn.Linear(2*5120, window_size)
        
        self.lora_tuning = lora_tuning
        if lora_tuning:
            if peft_dir:
                self.llm_model = PeftModel.from_pretrained(self.llm_model, peft_dir, is_trainable=True)
            else:
                peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
                self.llm_model = get_peft_model(self.llm_model, peft_config)
                self.llm_model.print_trainable_parameters()
        else:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False

        ## fixme: this is different from the original BLIP2
        self.eos_token_id = self.llm_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]
        self.pad_token_id = self.llm_tokenizer.pad_token_id

        # self.llm_proj = nn.Linear(
        #     self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        # )
        self.llm_proj = nn.Linear(
            300, self.llm_model.config.hidden_size
        )



        # print("Qformer.config.hidden_size")
        # print(self.Qformer.config.hidden_size)
        # print("llm_model.config.hidden_size")
        # print(self.llm_model.config.hidden_size)
        #
        ## fixme: no prompt yet
        self.prompt = prompt
        # prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")

        # print("prompt_tokens")
        # print(prompt_tokens)
        # exit(0)
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

    def forward(self, batch):


        graphs, text_tokens, prompt_lens = batch
        #
        # print("text_tokens.input_ids")
        # print(text_tokens.input_ids)

        # print("graphs.text:")
        # print(graphs.text)

        # device = graphs.device
        # print("text_tokens.input_ids")
        # print(text_tokens.input_ids)
        # print("self.llm_tokenizer.encode(graphs.text)")

        targets = []

        for i in range(len(graphs.text)):
            # targets.append(self.llm_tokenizer.encode(graphs.text[i], max_length=27, add_special_tokens=True,
            #                                 padding='max_length', truncation=True))
            # targets.append(self.llm_tokenizer(graphs.text[i],padding='max_length',max_length=27,
            #                                add_special_tokens=True,return_attention_mask=True, truncation=True)["input_ids"])
            targets.append(self.llm_tokenizer(graphs.text[i], padding='max_length', max_length=27,
                                              add_special_tokens=True, return_attention_mask=True, truncation=True)[
                               "input_ids"])

        # print(self.llm_tokenizer.batch_encode(graphs.text, max_length=200, add_special_tokens=True,
        #                              padding='max_length', truncation=True, return_tensors='pt'))


        # targets = self.llm_tokenizer.encode(graphs.text, max_length=200, add_special_tokens=True,
        #                              padding='max_length', truncation=True, return_tensors='pt')

        # print(input_ids)
        # exit(0)

        # print(self.llm_tokenizer.batch_decode(text_tokens.input_ids, skip_special_tokens=True))
        # exit(0)

        # print("text_tokens:")
        # print(text_tokens)

        # graph_embeds, graph_masks = self.graph_encoder(graphs)
        graph_embeds, graph_masks, _ = self.graph_encoder(graphs)

        # print("graph_embeds0")
        # print(graph_embeds)


        if not self.tune_gnn:
        # if self.tune_gnn:
            graph_embeds = graph_embeds.detach()
        graph_embeds = self.ln_graph(graph_embeds, graph_masks)




        # print("graph_embeds1")
        # print(graph_embeds)

        device = graph_embeds.device
        # query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        # print(query_tokens.shape)
        # # exit(0)
        #
        # print("query_tokens")
        # print(query_tokens)
        #
        # print("graph_embeds2.shape")
        # print(graph_embeds.shape)



        graph_embeds_new = []
        for i in range(len(query_tokens)):
            # print("###")
            # print(graphs.cand[i])
            if i == 0:
                # print(graph_embeds[i,:,2:4])
                # print(graph_embeds[i,[2,0],2:4])
                # exit(0)
                graph_embeds_new = torch.unsqueeze(graph_embeds[i,graphs.cand[i],:],dim=0)
                # print(graphs.cand[i])
                # print(graph_embeds_new.shape)
            else:
                graph_embeds_new = torch.cat((graph_embeds_new,torch.unsqueeze(graph_embeds[i,graphs.cand[i],:],dim=0)),0)
                # graph_embeds_new.co(graph_embeds[i,[0,2,0],:])
        graph_embeds = graph_embeds_new
        # graph_embeds = torch.Tensor(graph_embeds)
        # print(graph_embeds)
        #
        # print(graph_embeds.shape)

        # query_output = self.Qformer.bert(
        #     query_embeds=query_tokens,
        #     encoder_hidden_states=graph_embeds,
        #     encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
        #     return_dict=True,
        # )
        #
        # print("query_output.last_hidden_state.shape")
        # print(query_output.last_hidden_state.shape)
        #
        # exit(0)





        # inputs_llm = self.llm_proj(query_output.last_hidden_state)
        inputs_llm = self.llm_proj(graph_embeds)


        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(device)

        # print(inputs_llm.shape)
        # print(atts_llm.shape)
        # exit(0)



        ## add
        # targets = text_tokens.input_ids
        targets = torch.tensor(targets, dtype=torch.long).to(device)
        # targets = torch.unsqueeze(targets,2)

        # print(targets.shape)


        # exit(0)


        '''
        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
        )

        # print("targets a:")
        # print(targets)

        
        if self.prompt:
            # targets = mask_by_len(targets, prompt_lens, -100) # do not apply loss to the prompt
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt
            
        '''

        empty_targets = (
            ## 替换
            # torch.ones(atts_llm.size(), dtype=torch.long).to(device).fill_(-100)
            torch.ones(atts_llm.size(), dtype=torch.long).to(device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        # print(targets.shape)
        #
        # exit(0)

        # print("targets")
        # print(targets)

        # if self.lora_tuning:
        #     inputs_embeds = self.llm_model.model.get_decoder().embed_tokens(text_tokens.input_ids)
        # else:
        #     inputs_embeds = self.llm_model.model.decoder.embed_tokens(text_tokens.input_ids)

        # print("inputs_llm")
        # print(inputs_llm)
        #
        # print(inputs_llm.shape)
        # print("!!!")
        # print(graph_embeds[:,:, 2:4])




        # inputs_embeds = self.llm_model.get_input_embeddings()(text_tokens.input_ids)
        # # print(inputs_embeds.shape)
        # # inputs_embeds = torch.cat([inputs_llm, inputs_embeds[:,[0,1,2,3,4],:]], dim=1)
        # # attention_mask = torch.cat([atts_llm, text_tokens.attention_mask[:,[0,1,2,3,4]]], dim=1)
        # inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)



        # attention_mask = torch.cat([atts_llm, text_tokens.attention_mask], dim=1)
        attention_mask = atts_llm
        inputs_embeds = inputs_llm




        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            # return_dict=True,
            output_hidden_states=True,
            # labels=targets,
        )
        ## outputs.logits.shape: [batch_size=2, 29,32001]
        ## outputs.hidden_states[0].shape = [2, 29, 5120] -> [2, 2, 5120]
        ## len(outputs.hidden_states): 41   llama2-13b 有 40 层
        ## outputs.hidden_states
        # print(outputs.hidden_states[0].shape)



        print(graphs.text)
        output_flatten = outputs.hidden_states[0].reshape((len(query_tokens),-1))

        final = self.action_head(output_flatten)

        truth = to_label(graphs.text)
        truth_tensor = torch.tensor(truth, dtype=torch.long).to(device)

        criterion = nn.CrossEntropyLoss()


        loss = criterion(final, truth_tensor)

        # print(final.shape)
        # exit(0)

        # loss = outputs.loss




        my_action = []
        for i in range(len(query_tokens)):
            my_action.append(torch.argmax(final[i]))
        print(my_action)

        print("loss:")
        print(loss)



        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        #
        # print("targets:")
        # print(graphs.text)
        # print("predictions:")
        # outputs2 = self.llm_model.generate(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     do_sample=False,
        #     top_p=0.9,
        #     temperature=1,
        #     num_beams=5,
        #     # max_length=max_length,
        #     max_new_tokens=128,
        #     min_length=1,
        #     pad_token_id=self.pad_token_id,
        #     eos_token_id=self.eos_token_id,
        #     repetition_penalty=1,
        #     length_penalty=1,
        #     num_return_sequences=1,
        #     # use_cache=False,
        # )
        # output_text = self.llm_tokenizer.batch_decode(outputs2, skip_special_tokens=True)
        # output_text = [text.strip() for text in output_text]
        # print(output_text)

        if not self.is_training:
            graphs, text_tokens, prompt_lens = batch
            #
            # print("text_tokens.input_ids")
            # print(text_tokens.input_ids)

            # print("graphs.text:")
            # print(graphs.text)

            # device = graphs.device
            # print("text_tokens.input_ids")
            # print(text_tokens.input_ids)
            # print("self.llm_tokenizer.encode(graphs.text)")
            # print("x:")
            # print(graphs.x)

            xx = [[0 for _ in range(3)] for _ in range(link_num)]
            graphs.x = torch.tensor(xx, dtype=torch.long).to(device)
            graphs.edge_index, transformed_edge_num = transformed_edge_index(link_num, my_link)
            graphs.edge_index = graphs.edge_index.to(device)
            graphs.edge_attr = torch.tensor([[0, 0, 1] for _ in range(transformed_edge_num)], dtype=torch.long).to(device)
            graphs.batch = torch.tensor([0 for _ in range(link_num)], dtype=torch.long).to(device)

            # graph_embeds, graph_masks, _ = self.graph_encoder(graphs)
            ## graph_embeds.shape: torch.Size([1, 80, 300])

            allocated_capacity = [0 for i in range(link_num)]

            total_cost = 0
            ## three features: 已经占了多少比例（动态变化）；预计flow占多少比例（可以超过100%）；normalized ip cost
            ## third feature
            for j in range(link_num):
                graphs.x[j][2] = ip_cost[j]
                cc = my_level(graphs.x[j][2], np.min(ip_cost), np.max(ip_cost), level_c)
                graphs.x[j][2] = cc

            # second feature
            Gbase = nx.DiGraph()
            Gbase.add_nodes_from(my_node)
            edge_list = []
            for i in range(len(from_list)):
                edge_list.append((from_list[i], to_list[i]))
            Gbase.add_edges_from(edge_list)
            for i in range(flow_num):
                curr_flow = i
                path = nx.dijkstra_path(Gbase, source=my_flow["src"][curr_flow], target=my_flow["dst"][curr_flow])
                for node in range(len(path) - 1):
                    a = path[node]
                    b = path[node + 1]
                    for s in range(len(from_list)):
                        if a == from_list[s] and b == to_list[s]:
                            graphs.x[s][1] += my_flow["dmd"][curr_flow]
            b_list = []
            for i in range(link_num):
                aa = graphs.x[i][1].cpu()
                b_list.append(aa)
            b_max = np.max(b_list)

            ## Added
            for i in range(link_num):
                # if graphs.x[i][1] % mini_unit == 0:
                #     allocated_capacity[i] = graphs.x[i][1]
                # else:
                # allocated_capacity[i] = graphs.x[i][1] - (graphs.x[i][1]%mini_unit) + mini_unit
                # allocated_capacity[i] = graphs.x[i][1] - (graphs.x[i][1] % mini_unit)
                # allocated_capacity[i] = graphs.x[i][1]
                if graphs.x[i][1] >= max_fiber*capacity_per_fiber:
                    allocated_capacity[i] = max_fiber * capacity_per_fiber
                total_cost += allocated_capacity[i]*ip_cost[i]


            for i in range(link_num):
                bb = my_level(graphs.x[i][1], 0, b_max, level_b)
                graphs.x[i][1] = bb





            # for i in range(link_num):
            #     aa = my_level(allocated_capacity[i], 0, max_fiber * capacity_per_fiber, level_a)
            #     graphs.x[i][0] = aa


            cand_list = []
            ## our generation
            for i in range(link_num):
                ## 很可能会要activated
                if graphs.x[i][1] > 0:
                    cand_list.append(i)

            if test_mood:
                total_cost = 0

            action_cnt = 0

            while not my_solver_LP(my_flow, my_link, my_node, link_num, flow_num, dmd_max, allocated_capacity):

                ## candidate generation for each step
                ## no-filtering generation
                cand_list = []
                for i in range(link_num):
                    if allocated_capacity[i] < max_fiber * capacity_per_fiber:
                        cand_list.append(i)

                ## our generation

                # cand_set = set(cand_list)
                # curr_len = len(cand_list)
                # ccnt = 0
                # for i in range(curr_len):
                #     if allocated_capacity[cand_list[i]]==max_fiber*capacity_per_fiber:
                #         ccnt += 1
                #         cand_set.remove(cand_list[i])
                #         for j in range(transformed_edge_num):
                #             if (graphs.edge_index[0][j] == cand_list[i]) and (allocated_capacity[graphs.edge_index[1][j]] < max_fiber*capacity_per_fiber):
                #                 cand_set.add(graphs.edge_index[1][j].cpu().item())
                #             if (graphs.edge_index[1][j] == cand_list[i]) and (allocated_capacity[graphs.edge_index[0][j]] < max_fiber*capacity_per_fiber):
                #                 cand_set.add(graphs.edge_index[0][j].cpu().item())
                # if len(cand_set) == 0:
                #     for i in range(link_num):
                #         if allocated_capacity[i] < max_fiber * capacity_per_fiber:
                #             cand_set.add(i)
                # cand_list = list(cand_set)
                #
                # print("asset count: "+str(ccnt))
                # print(cand_list)
                # print("Length of cand_list "+ str(len(cand_list)))

                if test_mood:
                    cand_list = [i for i in range(link_num)]






                curr_ans = 0
                ## divide-and-conquer
                remaining_cand_list = cand_list
                while len(remaining_cand_list) > window_size:
                    group_size = 0
                    if len(remaining_cand_list) % window_size == 0:
                        group_size = (len(remaining_cand_list) // window_size)
                    else:
                        group_size = (len(remaining_cand_list) // window_size) + 1

                    new_remaining_list = []
                    for i in range(group_size):
                        if (i+1)*window_size <= len(remaining_cand_list):
                            curr_cand_list = remaining_cand_list[i*window_size:(i+1)*window_size]
                        else:
                            curr_cand_list = remaining_cand_list[i * window_size:]
                            random.seed(datetime.datetime.now())
                            extend_list = random.sample( curr_cand_list, (i+1)*window_size - len(remaining_cand_list))
                            curr_cand_list = curr_cand_list + extend_list

                        graph_embeds, graph_masks, _ = self.graph_encoder(graphs)

                        ## graph_embeds: tensor([[[a,s,d,...],[]]])

                        graph_embeds = graph_embeds.detach()
                        graph_embeds = self.ln_graph(graph_embeds, graph_masks)

                        # print("graph_embeds1")
                        # print(graph_embeds)

                        device = graph_embeds.device

                        # query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)

                        graph_embeds_new = []
                        for i in range(1):
                            graph_embeds_new = torch.unsqueeze(graph_embeds[i, curr_cand_list, :], dim=0)

                        graph_embeds = graph_embeds_new
                        inputs_llm = self.llm_proj(graph_embeds)
                        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(device)

                        attention_mask = atts_llm
                        inputs_embeds = inputs_llm

                        outputs = self.llm_model(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            # return_dict=True,
                            output_hidden_states=True,
                            # labels=targets,
                        )

                        # print(outputs.hidden_states[0].shape)
                        #
                        # exit(0)

                        output_flatten = outputs.hidden_states[0].reshape((1, -1))

                        final = self.action_head(output_flatten)


                        my_action = []
                        for i in range(1):
                            my_action.append(torch.argmax(final[i]))
                        my_action = my_action[0].cpu()


                        # curr_ans = curr_cand_list[to_answer(output_text[0],window_size)]
                        curr_ans = curr_cand_list[my_action]



                        new_remaining_list.append(curr_ans)

                        # print("Added!")
                        # print(len(new_remaining_list))

                    remaining_cand_list = new_remaining_list

                ## infer directly (i.e. len(remaining_cand_list) <= window_size)
                group_size = 1
                for i in range(group_size):
                    if (i + 1) * window_size <= len(remaining_cand_list):
                        curr_cand_list = remaining_cand_list[i * window_size:(i + 1) * window_size]
                    else:
                        curr_cand_list = remaining_cand_list[i * window_size:]
                        random.seed(datetime.datetime.now())
                        extend_list = random.sample(curr_cand_list, (i + 1) * window_size - len(remaining_cand_list))
                        curr_cand_list = curr_cand_list + extend_list

                    graph_embeds, graph_masks, _ = self.graph_encoder(graphs)

                    ## graph_embeds: tensor([[[a,s,d,...],[]]])

                    graph_embeds = graph_embeds.detach()
                    graph_embeds = self.ln_graph(graph_embeds, graph_masks)


                    device = graph_embeds.device

                    graph_embeds_new = []
                    for i in range(1):
                        graph_embeds_new = torch.unsqueeze(graph_embeds[i, curr_cand_list, :], dim=0)

                    graph_embeds = graph_embeds_new
                    inputs_llm = self.llm_proj(graph_embeds)
                    atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(device)
                    # inputs_embeds = self.llm_model.get_input_embeddings()(text_tokens.input_ids)
                    # inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                    # attention_mask = torch.cat([atts_llm, text_tokens.attention_mask], dim=1)
                    attention_mask = atts_llm
                    inputs_embeds = inputs_llm
                    outputs = self.llm_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        # return_dict=True,
                        output_hidden_states=True,
                        # labels=targets,
                    )
                    output_flatten = outputs.hidden_states[0].reshape((1, -1))
                    final = self.action_head(output_flatten)
                    my_action = []
                    for i in range(1):
                        my_action.append(torch.argmax(final[i]))
                    my_action = my_action[0].cpu()
                    # curr_ans = curr_cand_list[to_answer(output_text[0],window_size)]
                    curr_ans = curr_cand_list[my_action]

                    ## add capacity
                    allocated_capacity[curr_ans] += mini_unit
                    graphs.x[curr_ans][0] = allocated_capacity[curr_ans]
                    aa = my_level(graphs.x[curr_ans][0], 0, max_fiber * capacity_per_fiber, level_a)
                    graphs.x[curr_ans][0] = aa
                    print("Choose link "+ str(curr_ans))

                    total_cost += mini_unit * ip_cost[curr_ans]

                    action_cnt += 1

                if test_mood:
                    total_cost -= mini_unit * ip_cost[curr_ans]
                    total_cost += ip_cost[curr_ans]
                    if action_cnt == 5:
                        break

            f = open('cost.txt', 'a')  # 若是'wb'就表示写二进制文件
            f.write("total cost: "+ str(total_cost)+"\n")
            f.close()
        return {"loss": loss}





    @torch.no_grad()
    def generate(
        self,
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        graphs = samples['graphs']
        prompt_tokens = samples['prompt_tokens']
        # prompt_lens = samples['prompt_lens']
        with self.maybe_autocast():
            graph_embeds, graph_masks, _ = self.graph_encoder(graphs)
            graph_embeds = self.ln_graph(graph_embeds)

            # print(graph_embeds.shape)

            query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)

            # print(query_tokens.shape)
            #
            # exit(0)

            # query_output = self.Qformer.bert(
            #     query_embeds=query_tokens,
            #     encoder_hidden_states=graph_embeds,
            #     encoder_attention_mask=graph_masks,
            #     return_dict=True,
            # )

            device = graph_embeds.device

            graph_embeds_new = []
            for i in range(len(query_tokens)):
                if i == 0:
                    graph_embeds_new = torch.unsqueeze(graph_embeds[i, graphs.cand[i], :], dim=0)
                    # print(graphs.cand[i])
                    # print(graph_embeds_new.shape)
                else:
                    graph_embeds_new = torch.cat(
                        (graph_embeds_new, torch.unsqueeze(graph_embeds[i, graphs.cand[i], :], dim=0)), 0)
                    # graph_embeds_new.co(graph_embeds[i,[0,2,0],:])
            graph_embeds = graph_embeds_new
            inputs_llm = self.llm_proj(graph_embeds)

            # inputs_llm = self.llm_proj(query_output.last_hidden_state)
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long, device=device)

            attention_mask = torch.cat([atts_llm, prompt_tokens.attention_mask], dim=1)

            if False:
                if do_sample:
                    query_embeds = inputs_llm.repeat_interleave(num_captions, dim=0)
                    num_beams = 1
                else:
                    query_embeds = inputs_llm.repeat_interleave(num_beams, dim=0)

                outputs = self.llm_model.generate(
                    input_ids=prompt_tokens.input_ids,
                    query_embeds=query_embeds,
                    attention_mask=attention_mask,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_length,
                    min_length=min_length,
                    eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                )

                prompt_length = prompt_tokens.input_ids.shape[1]
                output_text = self.opt_tokenizer.batch_decode(
                    outputs[:, prompt_length:], skip_special_tokens=True
                )
            else:
                # inputs_embeds = self.llm_model.get_input_embeddings()(prompt_tokens.input_ids)
                # inputs_embeds = torch.cat([inputs_llm, inputs_embeds[:, [0, 1, 2, 3, 4], :]], dim=1)
                # attention_mask = torch.cat([atts_llm, prompt_tokens.attention_mask], dim=1)

                inputs_embeds = self.llm_model.get_input_embeddings()(prompt_tokens.input_ids)
                inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, prompt_tokens.attention_mask], dim=1)
                # inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                # attention_mask = torch.cat([atts_llm, prompt_tokens.attention_mask], dim=1)

                # ## ADD, ELMINATE LITERAL PROMPT
                # inputs_embeds = inputs_llm
                # attention_mask = atts_llm

                # inputs_embeds = inputs_llm
                # attention_mask = atts_llm
                #
                # inputs_embeds = torch.reshape(inputs_embeds, [len(query_tokens), -1])



                outputs = self.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    # max_length=max_length,
                    max_new_tokens=max_length,
                    min_length=min_length,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                    # use_cache=False,
                )
                # outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
                output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]

            print("########### output:")
            print(output_text)

            return output_text