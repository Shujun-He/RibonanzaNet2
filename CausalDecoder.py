from Network import MultiHeadAttention
import torch.nn as nn
import torch


def causal_mask_filled(seq_len: int, device=None):
    # Returns a [seq_len x seq_len] float mask with -inf in masked positions, 0 elsewhere
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, 0.0)
    if device:
        mask = mask.to(device)
    return mask

class SimpleStructureModule(nn.Module):

    def __init__(self, d_model, nhead, 
                 dim_feedforward, pairwise_dimension, dropout=0.1,
                 ):
        super(SimpleStructureModule, self).__init__()
        #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = MultiHeadAttention(d_model, nhead, d_model//nhead, d_model//nhead, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, d_model//nhead, d_model//nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        #self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        #self.dropout4 = nn.Dropout(dropout)

        self.pairwise2heads=nn.Linear(pairwise_dimension,nhead,bias=False)
        self.pairwise_norm=nn.LayerNorm(pairwise_dimension)

        self.distance2heads=nn.Linear(1,nhead,bias=False)
        #self.pairwise_norm=nn.LayerNorm(pairwise_dimension)

        self.activation = nn.GELU()

        
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(*inputs)
            return inputs
        return custom_forward

    def forward(self, input):
        tgt , src,  pairwise_features, pred_t, src_mask = input
        
        #src = src*src_mask.float().unsqueeze(-1)

        pairwise_bias=self.pairwise2heads(self.pairwise_norm(pairwise_features)).permute(0,3,1,2)

        
        distance_matrix=pred_t[:,None,:,:]-pred_t[:,:,None,:]
        distance_matrix=(distance_matrix**2).sum(-1).clip(2,37**2).sqrt()
        distance_matrix=distance_matrix[:,:,:,None]
        distance_bias=self.distance2heads(distance_matrix).permute(0,3,1,2)

        #print(pairwise_bias.shape,distance_bias.shape)

        pairwise_bias=pairwise_bias+distance_bias

        # print(tgt.shape)
        # print(pairwise_bias.shape)
        
        causal_mask=causal_mask_filled(tgt.shape[1],device=src.device)

        pairwise_bias=pairwise_bias+causal_mask[None,None,:,:]

        # print(tgt.shape,src.shape,pairwise_bias.shape,src_mask.shape)
        # exit()

        res=tgt
        tgt,attention_weights = self.self_attn(tgt, tgt, tgt, mask=pairwise_bias, src_mask=src_mask)
        tgt = res + self.dropout1(tgt)
        tgt = self.norm1(tgt)

        # print(tgt.shape,src.shape)
        # exit()

        res=tgt
        tgt,attention_weights = self.cross_attn(tgt, src, src, src_mask=None)
        tgt = res + self.dropout1(tgt)
        tgt = self.norm3(tgt)
        # print(tgt.shape)
        # exit()


        res=tgt
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = res + self.dropout2(tgt)
        tgt = self.norm2(tgt)


        return tgt

if __name__ == "__main__":

    #test
    seq_len=20
    seq_dim=256
    pairwise_dimension=64
    decoder=SimpleStructureModule(seq_dim, 8, seq_dim*4, pairwise_dimension).cuda()
    #print(decoder)

    input=[torch.randn(1,seq_len,seq_dim).cuda(),
        torch.randn(1,seq_len,seq_dim).cuda(),
        torch.randn(1,seq_len,seq_len,pairwise_dimension).cuda(),
        torch.randn(seq_len,3).cuda(),
        torch.ones(1,seq_len).cuda()]
    input=[i.cuda() for i in input]

    output=decoder(input)