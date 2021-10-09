import torch
import torch.nn as nn
from transformers.configuration_bert import BertConfig
from transformers.modeling_albert import AlbertModel
import torch.nn.functional as F
import math

class DualReader(nn.Module):
    def __init__(self,args,n_op, n_domain, update_id,ans_vocab,slot_mm,turn=2):
        super(DualReader,self).__init__()
        bert_config = BertConfig.from_pretrained(args.model_name_or_path + "config.json")
        args.slot_size=30
        args.ans_size=200
        args.hidden_size=bert_config.hidden_size
        args.n_slot = 30
        self.n_slot=30
        self.args=args
        self.slot_mm=slot_mm
        self.turn=turn
        self.albert = AlbertModel.from_pretrained(args.model_name_or_path+"pytorch_model.bin", config=bert_config)
        self.albert.resize_token_embeddings(args.vocab_size)
        self.decoder = Decoder(args, 500)
        self.input_drop=nn.Dropout(p=0.5)

        smask=ans_vocab.sum(dim=-1).eq(0).long()
        smask=slot_mm.long().mm(smask)
        self.slot_mm=nn.Parameter(slot_mm,requires_grad=False)
        self.slot_ans_mask=nn.Parameter(smask,requires_grad=False)
        self.ans_vocab = nn.Parameter(torch.FloatTensor(ans_vocab.size(0), ans_vocab.size(1), args.hidden_size),
                                      requires_grad=True)
        self.max_ans_size = ans_vocab.size(-1)
        self.slot_ans_size = ans_vocab.size(1)
        self.eslots = ans_vocab.size(0)
        self.ans_bias = nn.Parameter(torch.FloatTensor(ans_vocab.size(0), ans_vocab.size(1), 1),requires_grad=True)
        self.pos_weight=nn.Parameter(torch.FloatTensor([1]),requires_grad=True)
        self.pos_bias = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
        self.hidden_size = bert_config.hidden_size
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.action_cls = nn.Linear(self.hidden_size, n_op)
        if turn!=2:
            self.has_ans1 = nn.Linear(self.hidden_size, 2)
            # self.has_ans1_global = nn.Parameter(torch.FloatTensor(bert_config.hidden_size, 30, 2),requires_grad=True)
            # self.has_ans1_local = nn.Linear(bert_config.hidden_size, 2)
        self.domain_cls = nn.Linear(self.hidden_size, n_domain)
        self.n_op = n_op
        self.n_domain = n_domain
        self.update_id = update_id
        self.W_Q=nn.Linear(self.hidden_size,self.hidden_size)
        self.start_output=nn.Linear(self.hidden_size,self.hidden_size)
        self.end_output=nn.Linear(self.hidden_size,self.hidden_size)

        torch.nn.init.xavier_normal_(self.ans_bias)
        torch.nn.init.xavier_normal_(self.ans_vocab)
        self.layernorm=torch.nn.LayerNorm(self.hidden_size)

    def init_ans_vocab(self,ans_vocab):
        slot_ans_size = ans_vocab.size(1)
        init_vocab = nn.Parameter(torch.FloatTensor(self.args.slot_size, slot_ans_size, self.args.hidden_size),
                                       requires_grad=True)
        self.max_ans_size=ans_vocab.size(-1)
        self.slot_ans_size=ans_vocab.size(1)
        self.eslots=ans_vocab.size(0)
        ans_vocab=ans_vocab.reshape((-1,self.max_ans_size))
        attention_mask=(ans_vocab!=0)
        token_type_ids=torch.zeros_like(ans_vocab)
        ans_vocab_batches=ans_vocab.split(10)
        attention_mask=attention_mask.split(10)
        token_type_ids=token_type_ids.split(10)

        ans_vocab_encoded=[]
        for i,ans_batch in enumerate(ans_vocab_batches):
            batch_len = ans_batch.size(0)
            if ans_batch.sum()!=0:
                _, ans_batch_encoded=self.albert(input_ids=ans_batch,
                                          token_type_ids=token_type_ids[i],
                                          attention_mask=attention_mask[i])
            else:
                ans_batch_encoded=torch.zeros(batch_len,self.hidden_size)
            ans_vocab_encoded.append(ans_batch_encoded)
        ans_vocab_encoded=torch.cat(ans_vocab_encoded,dim=0)
        ans_vocab_encoded=ans_vocab_encoded.reshape(-1,slot_ans_size,self.hidden_size)
        init_vocab.data.copy_(ans_vocab_encoded)


    def forward(self, input_ids, token_type_ids,
                state_positions, attention_mask,slot_mask,
                max_value,op_ids=None, max_update=None,slot_ans_ids=None):
        enc_outputs = self.albert(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask)
        sequence_output, pooled_output = enc_outputs[:2]
        state_pos = state_positions[:, :, None].expand(-1, -1, sequence_output.size(-1))
        state_output = torch.gather(sequence_output, 1, state_pos)
        sequence_output=self.input_drop(sequence_output)
        seq_len=sequence_output.size(1)
        # batch_size=sequence_output.size(0)
        # state_mask=(torch.linspace(0,seq_len-1,seq_len).unsqueeze(0).repeat(batch_size,1).long().cuda()>=(state_pos[:,0,0].unsqueeze(-1)))

        state_output=state_output.view(-1,1,self.hidden_size)
        start_output=self.start_output(sequence_output)
        end_output=self.end_output(sequence_output)
        start_output=self.layernorm(start_output)
        end_output=self.layernorm(end_output)
        start_atten_m = state_output.view(-1,self.args.n_slot,self.hidden_size).bmm(start_output.transpose(-1,-2)).view(-1,self.args.n_slot,seq_len)/math.sqrt(self.hidden_size)
        end_atten_m = state_output.view(-1,self.args.n_slot,self.hidden_size).bmm(end_output.transpose(-1,-2)).view(-1,self.args.n_slot,seq_len)/math.sqrt(self.hidden_size)
        start_logits =start_atten_m.masked_fill(slot_mask.unsqueeze(1)==0,-1e9)
        end_logits = end_atten_m.masked_fill(slot_mask.unsqueeze(1)==0,-1e9)
        if self.turn==2:
            start_logits_softmax =F.softmax(start_logits[:,:,1:],dim=-1)
            end_logits_softmax =F.softmax(end_logits[:,:,1:],dim=-1)
        else:
            start_logits_softmax = F.softmax(start_logits, dim=-1)
            end_logits_softmax = F.softmax(end_logits, dim=-1)
        ques_attn=F.softmax((sequence_output.repeat(self.args.n_slot,1,1).bmm(state_output.transpose(-1,-2))/math.sqrt(self.hidden_size)).masked_fill(slot_mask.repeat(self.args.n_slot,1).unsqueeze(-1)==0,-1e9),dim=1)
        sequence_pool_output=ques_attn.transpose(-1,-2).bmm(sequence_output.repeat(self.args.n_slot,1,1)).squeeze()
        if self.turn==2:
            has_ans=torch.Tensor([1]).cuda()
        else:
            has_ans=self.has_ans1(sequence_pool_output).view(-1,self.args.n_slot,2)

        #category answer generating
        sequence_pool_output=sequence_pool_output.view(-1,self.args.n_slot,self.hidden_size)
        category_ans=sequence_pool_output.transpose(0,1).bmm(self.slot_mm.mm(self.ans_vocab.view(self.eslots,-1)).view(self.n_slot,self.slot_ans_size,-1).transpose(-1,-2))+self.slot_mm.mm(self.ans_bias.squeeze()).unsqueeze(1)
        category_ans=category_ans.transpose(0,1)
        category_ans=category_ans.masked_fill((self.slot_ans_mask==1).unsqueeze(0),-1e9)
        category_ans_softmax=F.softmax(category_ans,dim=-1)

        return start_logits_softmax, end_logits_softmax, has_ans,category_ans_softmax, start_logits, end_logits, category_ans

class Decoder(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(Decoder, self).__init__()
        self.pad_idx = 0
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        #self.embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.pad_idx)
        #self.embed.weight = bert_model_embedding_weights
        self.ans_vocab=nn.Parameter(torch.FloatTensor(config.slot_size,config.ans_size,config.hidden_size),requires_grad=True)
        self.ans_bias=nn.Parameter(torch.FloatTensor(config.slot_size,config.ans_size,1))
        self.gru = nn.GRU(config.hidden_size, config.hidden_size, 1, batch_first=True)
        self.w_gen = nn.Linear(config.hidden_size*3, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(config.dropout)

        # for n, p in self.gru.named_parameters():
        #     if 'weight' in n:
        #         p.data.normal_(mean=0.0, std=config.initializer_range)

    def forward(self, x, decoder_input, encoder_output, hidden, max_len, teacher=None):
        mask = x.eq(self.pad_idx)
        batch_size, n_update, _ = decoder_input.size()  # B,J',5 # long
        state_in = decoder_input
        all_point_outputs = torch.zeros(n_update, batch_size, max_len, self.vocab_size).to(x.device)
        result_dict = {}
        for j in range(n_update):
            w = state_in[:, j].unsqueeze(1)  # B,1,D
            slot_value = []
            for k in range(max_len):
                w = self.dropout(w)
                _, hidden = self.gru(w, hidden)  # 1,B,D
                # B,T,D * B,D,1 => B,T
                attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
                attn_e = attn_e.squeeze(-1).masked_fill(mask, -1e9)
                attn_history = nn.functional.softmax(attn_e, -1)  # B,T

                # B,D * D,V => B,V
                attn_v = torch.matmul(hidden.squeeze(0), self.embed.weight.transpose(0, 1))  # B,V
                attn_vocab = nn.functional.softmax(attn_v, -1)

                # B,1,T * B,T,D => B,1,D
                context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D

                p_gen = self.sigmoid(self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1)))  # B,1
                p_gen = p_gen.squeeze(-1)

                p_context_ptr = torch.zeros_like(attn_vocab).to(x.device)
                p_context_ptr.scatter_add_(1, x, attn_history)  # copy B,V
                p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
                _, w_idx = p_final.max(-1)
                slot_value.append([ww.tolist() for ww in w_idx])
                if teacher is not None:
                    w = self.embed(teacher[:, j, k]).unsqueeze(1)
                else:
                    w = self.embed(w_idx).unsqueeze(1)  # B,1,D
                all_point_outputs[j, :, k, :] = p_final

        return all_point_outputs.transpose(0, 1)

