import torch.nn as nn
import torch

class OptimizedDecoderLayer(nn.Module):
    def __init__(self, d_model, memory_dim, nhead, pair_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout = dropout, batch_first=True)
        self.cross_concat = CrossConcat(d_model, memory_dim, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.pair2heads = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, nhead))

    #def forward(self, tgt, memory, tgt_mask=None, past_key_value=None, use_cache=False):
    def forward(self, input):
        tgt, memory, pairwise_features, tgt_mask, past_key_value, use_cache = input

        Nr=tgt.shape[0]//pairwise_features.shape[0]

        # Self-attention
        # print(tgt.shape)
        # print(memory.shape)
        # print(pairwise_features.shape)
        # exit()
        if past_key_value is not None:
            tgt_res = tgt
            tgt_q = tgt[:,-1,None]
            tgt_k, tgt_v = past_key_value
            tgt_k = torch.cat([tgt_k, tgt_q], dim=1)
            tgt_v = torch.cat([tgt_v, tgt_q], dim=1)
            tgt_mask = None #no masking when doing kv cached decoding
            # print(tgt_q.shape)
            # print(tgt_k.shape)
            # print(tgt_v.shape)
            # exit()
        else:
            tgt_q, tgt_k, tgt_v = tgt, tgt, tgt


        attn_bias = self.pair2heads(pairwise_features).permute(0,3,1,2)#.squeeze()
        # print(attn_bias.shape)
        # exit()
        attn_bias = attn_bias+tgt_mask[None,None,:]
        attn_bias = attn_bias.unsqueeze(0)
        
        attn_bias=attn_bias.expand(Nr,-1,-1,-1,-1).transpose(1,0).reshape(-1,attn_bias.shape[2],attn_bias.shape[3],attn_bias.shape[4])
        # print(attn_bias.shape)
        # exit()
        attn_bias = attn_bias.reshape(-1,attn_bias.shape[2],attn_bias.shape[3])
        
        # print(tgt_q.shape)
        # print(attn_bias.shape)
        # #print(attn_bias.shape)
        # exit()
        self_attn_output, _ = self.self_attn(tgt_q, tgt_k, tgt_v, attn_mask=attn_bias)
        


        tgt_q = self.norm1(tgt_q + self_attn_output)
        

        mem_k, mem_v = memory, memory
        
        #cross_attn_output, _ = self.multihead_attn(tgt_q, mem_k, mem_v)
        # print(tgt_q.shape)
        # print(mem_k.shape)
        # exit()
        concat_product=self.cross_concat(tgt_q, memory)
        tgt_q = self.norm2(tgt_q + concat_product)
        
        # Feed-forward network
        ff_output = self.ffn(tgt_q)
        tgt_q = self.norm3(tgt_q + ff_output)

        if use_cache:
            tgt_q = torch.cat([tgt[:,:-1],tgt_q],1)
            # print(tgt.shape)
            # print(tgt_q.shape)
        else:
            pass
        
        #exit()
        if use_cache:
            return tgt_q, (tgt_k, tgt_v)
        else:
            return tgt_q, None


class CrossConcat(nn.Module):
    def __init__(self, d_model, memory_dim, dropout):
        super().__init__()
        self.linear=nn.Linear(d_model+memory_dim,d_model)
        self.dropout=nn.Dropout(dropout)
        self.gate=nn.Linear(d_model+memory_dim,d_model)


    def forward(self, x1, x2):
        x=torch.cat([x1,x2],dim=-1)
        g=self.gate(x).sigmoid()

        out=self.linear(x)
        out=self.dropout(out)
        
        return out*g