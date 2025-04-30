import sys

sys.path.append("/kaggle/input/ribonanzanet2d-final")

import torch.nn as nn
from Network import RibonanzaNet, MultiHeadAttention
import yaml
import torch

class SimpleStructureModule(nn.Module):

    def __init__(self, d_model, nhead, 
                 dim_feedforward, pairwise_dimension, dropout=0.1,
                 ):
        super(SimpleStructureModule, self).__init__()
        #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = MultiHeadAttention(d_model, nhead, d_model//nhead, d_model//nhead, dropout=dropout)


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

    def forward(self, src , pairwise_features, pred_t, src_mask=None, return_aw=False, use_gradient_checkpoint=False):
        
        #src = src*src_mask.float().unsqueeze(-1)

        pairwise_bias=self.pairwise2heads(self.pairwise_norm(pairwise_features)).permute(0,3,1,2)

        
        distance_matrix=pred_t[None,:,:]-pred_t[:,None,:]
        distance_matrix=(distance_matrix**2).sum(-1).clip(2,37**2).sqrt()
        distance_matrix=distance_matrix[None,:,:,None]
        #print(distance_matrix.shape)
        
        distance_bias=self.distance2heads(distance_matrix).permute(0,3,1,2)
        #print(distance_bias.shape)
        #print(pairwise_bias.shape)
        #exit()
        #.permute(0,3,1,2)

                    
        
        pairwise_bias=pairwise_bias+distance_bias

        #print(src.shape)
        src2,attention_weights = self.self_attn(src, src, src, mask=pairwise_bias, src_mask=src_mask)
        

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        if return_aw:
            return src,attention_weights
        else:
            return src



class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)



class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config, pretrained=False):
        config.dropout=0.1
        config.use_grad_checkpoint=False
        super(finetuned_RibonanzaNet, self).__init__(config)
        if pretrained:
            self.load_state_dict(torch.load("/kaggle/input/ribonanzanet-weights/RibonanzaNet.pt",map_location='cpu'))
        # self.ct_predictor=nn.Sequential(nn.Linear(64,256),
        #                                 nn.ReLU(),
        #                                 nn.Linear(256,64),
        #                                 nn.ReLU(),
        #                                 nn.Linear(64,1)) 
        self.dropout=nn.Dropout(0.0)

        self.structure_module=SimpleStructureModule(d_model=256, nhead=8, 
                 dim_feedforward=1024, pairwise_dimension=64)
        
        self.xyz_predictor=nn.Linear(256,3)


    
    def forward(self,src):
        
        #with torch.no_grad():
        sequence_features, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))
        
        xyzs=[]
        xyz=torch.zeros(sequence_features.shape[1],3).cuda().float()
        #print(xyz.shape)
        #xyz=self.xyz_predictor(sequence_features)

        for i in range(9):
            sequence_features=self.structure_module(sequence_features,pairwise_features,xyz)
            xyz=xyz+self.xyz_predictor(sequence_features).squeeze(0)
            xyzs.append(xyz)
            
        
        return xyzs

model=finetuned_RibonanzaNet(load_config_from_yaml("pairwise.yaml"),pretrained=False).cuda()

model(torch.ones(1,10).long().cuda())