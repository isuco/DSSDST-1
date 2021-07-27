

import numpy as np
import json
from torch.utils.data import Dataset
from utils import constant
import torch
import random
import re
from copy import deepcopy
import concurrent.futures as fu
from functools import reduce
import difflib
from .fix_label import fix_general_label_error
import torch.nn.functional as F

flatten = lambda x: [i for s in x for i in s]
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
domain2id = {d: i for i, d in enumerate(EXPERIMENT_DOMAINS)}
OP_SET = {
    '2': {'update': 0, 'carryover': 1},
    '3-1': {'update': 0, 'carryover': 1, 'dontcare': 2},
    '3-2': {'update': 0, 'carryover': 1, 'delete': 2},
    '4': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3},
    '6': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3, 'yes': 4, 'no': 5}
}
track_slots=[ "attraction-area",
  "attraction-name",
  "attraction-type",
  "hotel-area",
  "hotel-bookday",
  "hotel-bookpeople",
  "hotel-bookstay",
  "hotel-internet",
  "hotel-name",
  "hotel-parking",
  "hotel-pricerange",
  "hotel-stars",
  "hotel-type",
  "restaurant-area",
  "restaurant-bookday",
  "restaurant-bookpeople",
  "restaurant-booktime",
  "restaurant-food",
  "restaurant-name",
  "restaurant-pricerange",
  "taxi-arriveby",
  "taxi-departure",
  "taxi-destination",
  "taxi-leaveat",
  "train-arriveby",
  "train-bookpeople",
  "train-day",
  "train-departure",
  "train-destination",
  "train-leaveat"]

def map_state_to_ids(slot_state,slot_meta,slot_ans):
    keys = list(slot_state.keys())
    slot_ans_idx = [-1] * len(slot_meta)
    for k in keys:
        if k[:8]=='hospital' or k[:5]=='polic':
            continue
        v = slot_state[k]
        if v==[]:
            continue
        v_list = slot_meta[k]['db']
        st=slot_meta[k]['type']
        v=v[0]
        if not st:
            v_idx=-1
        else:
            if v in v_list:
                v_idx = v_list.index(v)
            else:
                v_idx = find_value_idx(v, v_list)
                slot_state[k]=v_list[v_idx]
        for i,z in enumerate(slot_ans):
            if z['name']==k:
                slot_ans_idx[i] = v_idx
                break

    return slot_ans_idx

def make_turn_label(slot_meta, last_dialog_state, turn_dialog_state,
                    tokenizer, slot_ans=None,op_code='4', dynamic=False,turn=0):
    if dynamic:
        gold_state = turn_dialog_state
        turn_dialog_state = {}
        for x in gold_state:
            s = x.split('-')
            k = '-'.join(s[:2])
            turn_dialog_state[k] = s[2]
    op_labels = ['carryover'] * len(slot_meta)
    generate_idx=[[0,0]]*len(slot_meta)
    generate_y=[]
    for s in slot_meta:
        generate_y.append([])
    keys = list(turn_dialog_state.keys())
    slot_ans_idx=[-1]*len(slot_meta)
    for k in keys:
        if k[:8]=='hospital' or k[:5]=='polic':
            continue
        v = turn_dialog_state[k]
        for z,sa in enumerate(slot_ans):
            if sa['name']==k:
                k_idx=z
                break
        iscate=slot_meta[k]['type']
        if iscate and v!=[]:
            vv=v[0]
            v_list=slot_meta[k]['db']
            if vv in v_list:
                v_idx=v_list.index(vv)
            else:
                v_idx=find_value_idx(vv,v_list)
                turn_dialog_state[k]=v_list[v_idx]
        else:
            v_idx=-1
        slot_ans_idx[k_idx]=v_idx
        if v==['none']:
            turn_dialog_state[k]=[]
        vv = last_dialog_state.get(k)
        try:
            idx = k_idx
            if vv != v:
                op_labels[idx] = 'update'
                s_ans=[tokenizer.tokenize(sv) + ['[EOS]'] for sv in v]
                if ((v==['none'] or v==[]) and vv!=[]):
                    slot_ans_idx[idx] = len(slot_ans[idx]['db']) - 1
                    generate_y[idx] = ['[SEP]']
                    generate_idx[idx] = []
                else:
                    generate_y[idx] = s_ans
                    generate_idx[idx]=[]
            else:
                op_labels[idx]='carryover'
        except ValueError:
            continue
    # for k in last_dialog_state.keys():
    #     if k not in keys:
    #         for z, sa in enumerate(slot_ans):
    #             if sa['name'] == k:
    #                 idx = z
    #                 break
    #         slot_ans_idx[idx] = len(slot_ans[k]) - 1
    #         generate_y.append(['[SEP]'])
    #         generate_idx[idx] = []
    #         op_labels[idx]='update'
    # if turn==1 or turn==0:
    #     for i,val in enumerate(slot_ans_idx):
    #         if val==-1:
    #             slot_ans_idx[i]=len(slot_meta[slot_ans[i]['name']]['db'])-2
    # for k, v in last_dialog_state.items():
    #     vv = turn_dialog_state.get(k)
    #     for si,sa in enumerate(slot_ans):
    #         if sa['name']==k:
    #             idx=si
    #             break
    #     try:
    #         if vv==[] and:
    #             if OP_SET[op_code].get('delete') is not None:
    #                 op_labels[idx] = 'delete'
    #             else:
    #                 slot_ans_idx[idx] = len(slot_ans[idx]['db']) - 1
    #                 generate_idx[idx] = [-1, -1]
    #                 op_labels[idx] = 'update'
    #                 # generate_y.append([['[NULL]', '[EOS]'], idx])
    #     except ValueError:
    #         continue
    gold_state = [str(k) + '-' + str(v[0]) if v!=[] else str(k)+'-[]' for k, v in turn_dialog_state.items()]

    if dynamic:
        op2id = OP_SET[op_code]
        generate_y = [tokenizer.convert_tokens_to_ids(y) for y in generate_y]
        op_labels = [op2id[i] for i in op_labels]

    return op_labels, generate_y, gold_state,generate_idx,slot_ans_idx

def find_value_idx(v,v_list):
    if v=='dontcare':
        return v_list.index("[dontcare]")
    elif v=='wartworth':
        return v_list.index("warkworth house")
    else:
        for idx,label_v in enumerate(v_list):
            v=v.replace(" ","")
            v=v.replace("2","two")
            l_v=label_v.replace(" ","")
            l_v=l_v.replace("\'","")
            if v in l_v:
                return idx
    max_similar=0
    max_idx=-1
    for idx,label_v in enumerate(v_list):
        similarity=difflib.SequenceMatcher(None, v, label_v).quick_ratio()
        if similarity>max_similar:
            max_similar=similarity
            max_idx=idx
    return max_idx

def postprocessing(slot_meta, ops, last_dialog_state,
                   generated, tokenizer, op_code, gold_gen={}):
    gid = 0
    for st, op in zip(slot_meta, ops):
        if op == 'dontcare' and OP_SET[op_code].get('dontcare') is not None:
            last_dialog_state[st] = 'dontcare'
        elif op == 'yes' and OP_SET[op_code].get('yes') is not None:
            last_dialog_state[st] = 'yes'
        elif op == 'no' and OP_SET[op_code].get('no') is not None:
            last_dialog_state[st] = 'no'
        elif op == 'delete' and last_dialog_state.get(st) and OP_SET[op_code].get('delete') is not None:
            last_dialog_state.pop(st)
        elif op == 'update':
            g = tokenizer.convert_ids_to_tokens(generated[gid])
            gen = []
            for gg in g:
                if gg == '[EOS]':
                    break
                gen.append(gg)
            gen = ' '.join(gen).replace(' ##', '')
            gid += 1
            gen = gen.replace(' : ', ':').replace('##', '')
            if gold_gen and gold_gen.get(st) and gold_gen[st] not in ['dontcare']:
                gen = gold_gen[st]

            if gen == '[NULL]' and last_dialog_state.get(st) and not OP_SET[op_code].get('delete') is not None:
                last_dialog_state.pop(st)
            else:
                last_dialog_state[st] = gen

    return generated, last_dialog_state

def make_slot_meta(ontology,turn=0):
    meta = []
    change = {}
    idx = 0
    max_len = 0
    esm_ans=constant.ansvocab
    slot_map=constant.slot_map
    for ea in esm_ans:
        if turn < 2:
            ea.append("[noans]")
        if ea is not None:
            ea.append("[negans]")
            ea.append("[dontcare]")

    for i, k in enumerate(ontology):
        d=k['service_name']

        if d not in EXPERIMENT_DOMAINS:
            continue
        slots=k['slots']
        for s in slots:
            if s['name'] not in track_slots:
                continue
            if 'price' in s['name'] or 'leave' in s['name'] or 'arrive' in s['name']:
                s['name'] = s['name'].replace(' ', '')
            slot_dic={}
            slot_dic['type']=s['is_categorical']
            if slot_dic['type']:
                slot_dic['db']=esm_ans[slot_map[s['name']]]
            change[s['name']] = slot_dic

    return sorted(meta), change

global_tokenizer=None
global_slot_meta=None
global_n_history=None
global_max_seq_length=None
global_slot_ans=None
global_diag_level=None
global_op_code=None
global_pred_op=None
global_isfilter=False
global_turn=0

def process_dial_dict(dial_dict):
    datas=[]
    global global_max_seq_length, global_op_code, global_tokenizer, global_slot_ans, global_slot_ans, global_slot_meta, global_n_history, global_diag_level,global_pred_op,global_isfilter,global_turn
    dialog_history = []
    last_dialog_state = {}
    for sa in global_slot_ans:
        last_dialog_state[sa['name']]=[]

    last_uttr = ""
    for ti, turn in enumerate(dial_dict["dialogue"]):
        turn_domain = turn["domain"]
        if turn_domain not in EXPERIMENT_DOMAINS:
            continue
        turn_id = turn["turn_idx"]
        turn_uttr = (turn["system_transcript"] + ' ; ' + turn["transcript"]).strip()
        dialog_history.append(last_uttr)
        pop_state = []
        turn_dialog_state = fix_general_label_error(turn["belief_state"], False, global_slot_meta)
        last_uttr = turn_uttr
        sample_ids = dial_dict["dialogue_idx"] + "_" + str(turn_id)
        op_labels, generate_y, gold_state, generate_idx, slot_ans_idx = make_turn_label(global_slot_meta,
                                                                                        last_dialog_state,
                                                                                        turn_dialog_state,
                                                                                        global_tokenizer,
                                                                                        slot_ans=global_slot_ans,
                                                                                        op_code=global_op_code,
                                                                                        turn=global_turn,
                                                                        dynamic=False)
        if (ti + 1) == len(dial_dict["dialogue"]):
            is_last_turn = True
        else:
            is_last_turn = False
        #last_dialog_state=map_state_to_id(last_dialog_state,global_slot_meta,global_slot_ans)

        gold_state_idx=map_state_to_ids(turn_dialog_state,global_slot_meta,global_slot_ans)
        turn_uttr = fixutter(turn_uttr, turn_dialog_state)
        last_uttr = turn_uttr

        if global_pred_op is not None:
            pred_op=np.array(global_pred_op[sample_ids])
        else:
            pred_op=[]
        isdrop=False

        if global_turn==1:
            isdrop = global_isfilter and (all(op == 1 for op in pred_op.argmax(axis=-1)))
        if global_turn==2:
            isdrop = global_isfilter and (all(op=='carryover' for op in op_labels))
        if not isdrop:
            instance = TrainingInstance(dial_dict["dialogue_idx"], turn_domain,
                                        turn_id, turn_uttr, ' '.join(dialog_history[-global_n_history:]),
                                        last_dialog_state, op_labels,pred_op,
                                        generate_y, generate_idx, gold_state,gold_state_idx, global_max_seq_length, global_slot_meta,
                                        is_last_turn, slot_ans_idx, op_code=global_op_code)
            instance.make_instance(global_tokenizer,turn=global_turn)

            datas.append(instance)
        last_dialog_state = turn_dialog_state
    return datas

def fixutter(utter,tstate):
    state={}
    for si in tstate.keys():
        if tstate[si]!=[]:
            state[si]=tstate[si]
    similar_str={}
    if state=={}:
        return utter
    for ui in range(len(utter)):
        if utter[ui] == " ":
            continue
        for sv in state.values():
            s=sv[0]
            i = 0
            j = 0
            canstr=""
            ismatch= (ui+len(s) < len(utter))
            if ismatch:
                while (i < len(s) and ui+j < len(utter)):
                    if s[i]==" " or s[i]=="'":
                        i+=1
                        continue
                    canstr+=utter[ui+j]
                    if utter[j+ui]==" " or utter[j+ui]=="'":
                        j+=1
                        continue
                    else:
                        if s[i]!=utter[ui+j]:
                            ismatch=False
                            break
                        i+=1
                        j+=1
            if ismatch:
                similar_str[canstr]=s
    for si in similar_str.keys():
        if si!='':
            utter=utter.replace(si,similar_str[si])
    return utter

def prepare_dataset(data_path, tokenizer, slot_meta,
                    n_history, max_seq_length, slot_ans=None,diag_level=False, op_code='4',op_data_path=None,isfilter=True,turn=0):
    global global_max_seq_length,global_op_code,global_tokenizer,global_slot_ans,global_slot_ans,global_slot_meta,global_n_history,global_diag_level,global_pred_op,global_isfilter,global_turn
    global_tokenizer=tokenizer
    global_diag_level=diag_level
    global_max_seq_length=max_seq_length
    global_n_history=n_history
    global_op_code=op_code
    global_slot_meta=slot_meta
    global_slot_ans=slot_ans
    global_isfilter=isfilter
    global_turn=turn
    if op_data_path is not None:
        with open(op_data_path,'r') as f:
            global_pred_op=json.load(f)
    datas=[]
    dials = json.load(open(data_path))
    # for d in dials:
    #     datas+=process_dial_dict(d)
    with fu.ProcessPoolExecutor() as excutor:
        datas=list(excutor.map(process_dial_dict, dials))
    datas=reduce(lambda x,y:x+y,datas)
    return datas,[],[]



class TrainingInstance:
    def __init__(self, ID,
                 turn_domain,
                 turn_id,
                 turn_utter,
                 dialog_history,
                 last_dialog_state,
                 op_labels,
                 pred_op,
                 generate_y,
                 generate_idx,
                 gold_state,
                 gold_state_idx,
                 max_seq_length,
                 slot_meta,
                 is_last_turn,
                 slot_ans_ids,
                 op_code='4'):
        self.id = str(ID)+"_"+str(turn_id)
        self.turn_domain = turn_domain
        self.turn_id = turn_id
        self.turn_utter = turn_utter
        self.dialog_history = dialog_history
        self.last_dialog_state = last_dialog_state
        self.gold_p_state = last_dialog_state
        self.gold_state_idx=gold_state_idx
        self.generate_y = generate_y
        self.generate_idx = generate_idx
        self.slot_ans_ids=slot_ans_ids
        self.op_labels = op_labels
        self.pred_op=pred_op
        self.gold_state = gold_state
        self.max_seq_length = max_seq_length
        self.slot_meta = slot_meta
        self.is_last_turn = is_last_turn
        self.op2id = OP_SET[op_code]

    def shuffle_state(self, rng, slot_meta=None):
        #don't fix
        new_y = []
        gid = 0
        for idx, aa in enumerate(self.op_labels):
            if aa == 'update':
                new_y.append(self.generate_y[gid])
                gid += 1
            else:
                new_y.append(["dummy"])
        if slot_meta is None:
            temp = list(zip(self.op_labels, self.slot_meta, new_y))
            rng.shuffle(temp)
        else:
            indices = list(range(len(slot_meta)))
            for idx, st in enumerate(slot_meta):
                indices[self.slot_meta.index(st)] = idx
            temp = list(zip(self.op_labels, self.slot_meta, new_y, indices))
            temp = sorted(temp, key=lambda x: x[-1])
        temp = list(zip(*temp))
        self.op_labels = list(temp[0])
        self.slot_meta = list(temp[1])
        self.generate_y = [yy for yy in temp[2] if yy != ["dummy"]]

    # def findidx(self,gen_ids,input_ids):
    #     for g in gen_ids:


    def make_instance(self, tokenizer,lack_ans=[],max_seq_length=None,
                      word_dropout=0., slot_token='[SLOT]',turn=0,eval_token=False):
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        state = []
        for s in self.slot_meta:
            state.append(slot_token)
            k = s.split('-')
            v = self.last_dialog_state.get(s)
            if v !=[]:
                if not eval_token:
                    v=v[0]
                if not self.slot_meta[s]['type'] and eval_token:
                    t = tokenizer.tokenize(' '.join(k))
                    t=t+['-']+v
                else:
                    k.extend(['-', v])
                    t=tokenizer.tokenize(' '.join(k))
            else:
                t = tokenizer.tokenize(' '.join(k))
                t.extend(['-', '[NULL]'])
            state.extend(t)

        #only use present utter

        avail_length_1 = max_seq_length - len(state) - 3

        if turn == 0 or turn == 1:

            diag_2 = tokenizer.tokenize(self.turn_utter)


            if len(diag_2) > avail_length_1:
                avail_length = len(diag_2) - avail_length_1
                diag_2 = diag_2[avail_length:]
            drop_mask = [0] + [1] * len(diag_2) + [0]
            diag_2 = ["[CLS]"] + diag_2 + ["[SEP]"]
            segment = [0] * len(diag_2)
            diag = diag_2
        else:
            diag_1 = tokenizer.tokenize(self.dialog_history)
            diag_2 = tokenizer.tokenize(self.turn_utter)
            avail_length = avail_length_1 - len(diag_2)

            if len(diag_1) > avail_length:  # truncated
                avail_length = len(diag_1) - avail_length
                diag_1 = diag_1[avail_length:]

            if len(diag_1) == 0 and len(diag_2) > avail_length_1:
                avail_length = len(diag_2) - avail_length_1
                diag_2 = diag_2[avail_length:]
            # if len(diag_2) > avail_length_1:
            #     avail_length = len(diag_2) - avail_length_1
            #     diag_2 = diag_2[avail_length:]
            drop_mask = [0] + [1] * len(diag_1) + [0] + [1] * len(diag_2) + [0]
            diag_1 = ["[CLS]"] + diag_1 + ["[SEP]"]
            diag_2 = diag_2 + ["[SEP]"]
            segment = [0] * len(diag_1) + [1] * len(diag_2)
            diag = diag_1 + diag_2


        # word dropout
        if word_dropout > 0.:
            drop_mask = np.array(drop_mask)
            word_drop = np.random.binomial(drop_mask.astype('int64'), word_dropout)
            diag = [w if word_drop[i] == 0 else '[UNK]' for i, w in enumerate(diag)]
        extra_ans=[]
        # for ai,ans in enumerate(lack_ans):
        #     extra_ans+=["[ANS]"]+ans
        input_ = diag +extra_ans+ state
        segment = segment + [1]*len(state)
        self.input_ = input_
        self.slot_mask=[1]*len(diag)
        self.segment_id = segment
        slot_position = []

        for i, t in enumerate(self.input_):
            if t == slot_token:
                slot_position.append(i)
        self.slot_position = slot_position

        input_mask = [1] * len(self.input_)
        self.input_id = tokenizer.convert_tokens_to_ids(self.input_)
        if len(input_mask) < max_seq_length:
            self.input_id = self.input_id + [0] * (max_seq_length-len(input_mask))
            self.segment_id = self.segment_id + [0] * (max_seq_length-len(input_mask))
            input_mask = input_mask + [0] * (max_seq_length-len(input_mask))
        self.slot_mask=self.slot_mask+[0]*(max_seq_length-len(self.slot_mask))
        self.input_mask = input_mask
        self.domain_id = domain2id[self.turn_domain]
        self.op_ids = [self.op2id[a] for a in self.op_labels]
        self.generate_ids = [[tokenizer.convert_tokens_to_ids(ans) for ans in y] for y in self.generate_y]
        self.start_idx,self.end_idx,lack_ans,span_mask=self.findidx(self.generate_ids,self.generate_idx,self.input_id,turn)
        self.start_position=[]
        self.end_position=[]
        # for gi,g in enumerate(self.generate_idx):
        #     s_p=[0]*max_seq_length
        #     e_p=[0]*max_seq_length
        #     s_p[g[0]]=1
        #     e_p[g[-1]]=1
        #     self.start_position.append(s_p)
        #     self.end_position.append(e_p)
        return lack_ans,span_mask

    def findidx(self,generate_y,generate_idx,inputs_idx,turn=0):
        lack_ans=[]
        span_mask=[1]*len(generate_idx)
        lastsep=-1
        firstsep=-1
        for i,ch in enumerate(inputs_idx):
            if ch==constant.ALBERT_SEP:
                if firstsep==-1:
                    firstsep=i
                lastsep=i

        for gi,value in enumerate(generate_y):
            if value==[]:
                continue
            elif value==[constant.ALBERT_SEP]:
                sep=lastsep
                generate_idx[gi] = [sep, sep]
            else:
                hasfound=False
                for i, t_id in enumerate(inputs_idx[:lastsep]):
                    for gans in value:
                        gans=gans[:-1]
                        g_len=len(gans)
                        if (gans==inputs_idx[i:i+g_len]):
                                #turn1 remove CLS
                            if turn==0 or turn==1:
                                generate_idx[gi]=[i,i+g_len-1]
                            elif turn==2:
                                # generate_idx[gi] = [i, i + g_len - 1]
                                generate_idx[gi] = [i, i + g_len-1]
                            hasfound=True
                            break
                    if hasfound:
                        break
        for gi,g in enumerate(generate_idx):
            if g==[]:
                generate_idx[gi]=[-1,-1]
        start_idx=[generate_idx[i][0] for i in range(len(generate_idx))]
        end_idx = [generate_idx[i][-1] for i in range(len(generate_idx))]
        return start_idx,end_idx,lack_ans,span_mask


class MultiWozDataset(Dataset):
    def __init__(self, data, tokenizer, slot_meta, max_seq_length,
                 ontology, word_dropout=0.1, turn=2):
        self.data = data
        self.len = len(data)
        self.tokenizer = tokenizer
        self.slot_meta = slot_meta
        self.max_seq_length = max_seq_length
        self.ontology = ontology
        self.word_dropout = word_dropout
        self.turn=turn

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.word_dropout > 0 or self.shuffle_state:
            self.data[idx].make_instance(self.tokenizer,
                                         word_dropout=self.word_dropout,turn=self.turn)
        return self.data[idx]

    def collate_fn(self, batch):
        input_ids = torch.tensor([f.input_id for f in batch], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_id for f in batch], dtype=torch.long)
        state_position_ids = torch.tensor([f.slot_position for f in batch], dtype=torch.long)
        op_ids = torch.tensor([f.op_ids for f in batch], dtype=torch.long)
        pred_op_ids=torch.tensor([f.pred_op for f in batch],dtype=torch.float)
        # pred_op_ids=F.softmax(pred_op_ids,dim=-1)
        slot_ans_ids=torch.tensor([f.slot_ans_ids for f in batch],dtype=torch.long)
        domain_ids = torch.tensor([f.domain_id for f in batch], dtype=torch.long)
        gen_ids = [b.generate_ids for b in batch]
        start_position=torch.tensor([b.start_position for b in batch],dtype=torch.long)
        end_position=torch.tensor([b.end_position for b in batch],dtype=torch.long)
        slot_mask=torch.tensor([f.slot_mask for f in batch],dtype=torch.long)
        start_idx=torch.tensor([f.start_idx for f in batch],dtype=torch.long)
        end_idx = torch.tensor([f.end_idx for f in batch], dtype=torch.long)
        update_len=[len(b) for b in gen_ids]
        value_len=[len(b) for b in flatten(gen_ids)]
        max_update = max(update_len) if len(update_len)!=0 else 0
        max_value = max(value_len) if len(value_len)!=0 else 0
        sid=[f.id for f in batch]
        for bid, b in enumerate(gen_ids):
            n_update = len(b)
            for idx, v in enumerate(b):
                b[idx] = v + [0] * (max_value - len(v))
            gen_ids[bid] = b + [[0] * max_value] * (max_update - n_update)
        # gen_ids = torch.tensor(gen_ids, dtype=torch.long)

        return input_ids, input_mask,slot_mask,segment_ids, state_position_ids, op_ids, pred_op_ids,domain_ids, gen_ids,start_position,end_position,max_value, max_update,slot_ans_ids,start_idx,end_idx,sid
