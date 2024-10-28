import math
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Sequence
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.utilities.model_summary import ModelSummary
import lightning
from lightning.fabric import Fabric
from dataclasses import dataclass
import einx

@dataclass
class ModelConfig:
	n_amino:int
	amino_dim:int
	n_heads:int
	n_layers:int
	expansion_factor:int

class FactorizedConv(lightning.LightningModule):

	def __init__(self,
			  in_dim,
			  feature,
			  kernel_size):
		
		super().__init__()
		
		self.conv = nn.Conv1d(in_dim,in_dim,kernel_size=kernel_size,
						groups=in_dim,padding=(kernel_size//2))
		self.dense = nn.Conv1d(in_dim,feature,1)
	
	def forward(self,x):

		x = einx.rearrange("b l d -> b d l",x)
		x = self.conv(x)
		x = self.dense(x)
		x = einx.rearrange("b d l -> b l d",x)
		return x
	
class MLP(lightning.LightningModule):

	def __init__(self,
			  in_dim:int,
			  features:list[int],
			  kernel_size:int):
		
		super().__init__()
		
		layers = []

		for feature in features:
			layers.append(FactorizedConv(in_dim,feature,kernel_size))
			in_dim = feature

		self.layers = nn.Sequential(*layers)
		self.ln = nn.LayerNorm(feature)

	def forward(self,x):

		for l in self.layers[:-1]:
			x = F.relu(l(x))
		x = self.ln(self.layers[-1](x))
		return x
	
def sin_pos_encode(x):
	embeddings = torch.zeros((x.shape)).to(x.device)
	batch,length,dim = x.shape
	position = torch.arange(0,length).to(x.device)
	omega = torch.exp(torch.arange(0,dim,2).to(x.device) * math.log(10000)/dim)
	embeddings[:,:,0::2] = torch.sin(einx.multiply("l,d -> l d",position,omega))
	embeddings[:,:,1::2] = torch.cos(einx.multiply("l,d -> l d",position,omega))

	return embeddings + x


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in torch < 1.8.0


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class MaskedMHA(lightning.LightningModule):

	def __init__(self,
			  hidden_dim:int,
			  heads:int):
		
		super().__init__()
		
		self.proj = nn.Linear(hidden_dim,hidden_dim*3)
		self.heads = heads
	
	def forward(self,x,mask):

		proj = self.proj(x)

		q,k,v = einx.rearrange("b l (k d) -> k b l d",proj,k=3)

		q = q * ((q.shape[-1] // self.heads) ** -0.5)
		attn = einx.dot("b q (h c), b k (h c) -> b q k h", q, k, h=self.heads)

		attn = einx.where("b k, b q k h,", mask, attn, -torch.inf)

		attn = einx.softmax("b q [k] h", attn)
		y = einx.dot("b q k h, b k (h c) -> b q (h c)", attn, v)

		return x + y
	
class ROPEMaskedMHA(lightning.LightningModule):

	def __init__(self,
			  hidden_dim:int,
			  heads:int):
		
		super().__init__()
		
		self.proj = nn.Linear(hidden_dim,hidden_dim*3)
		self.heads = heads
		self.rope = Rotary(hidden_dim//heads)

	
	def forward(self,x,mask):

		proj = self.proj(x)

		q,k,v = einx.rearrange("b l (k h d) -> k l b h d",proj,k=3,h=self.heads)

		cos,sin = self.rope(x)

		q,k = apply_rotary_pos_emb(q,k,cos,sin)

		q = q * ((q.shape[-1]) ** -0.5)
		attn = einx.dot("l1 b h c, l2 b h c -> b l1 l2 h", q, k, h=self.heads)

		attn = einx.where("b k, b q k h,", mask, attn, -torch.inf)

		attn = einx.softmax("b q [k] h", attn)
		y = einx.dot("b q k h, k b h c -> b q (h c)", attn, v)

		return x + y
	
class EncoderBlock(lightning.LightningModule):

	def __init__(self,
			  hidden_dim:int,
			  heads:int,
			  expansion_factor:int):
		
		super().__init__()
		
		self.mha = ROPEMaskedMHA(hidden_dim,heads)
		self.mlp = MLP(hidden_dim,[hidden_dim*expansion_factor,hidden_dim],kernel_size=3)
		self.mha_ln = nn.LayerNorm(hidden_dim)
		self.mlp_ln = nn.LayerNorm(hidden_dim)
		self.drop = nn.Dropout(0.1)

	def forward(self,x,mask):

		x = x + self.drop(self.mha_ln(self.mha(x,mask))) + self.drop(self.mlp_ln(self.mlp(x)))
		return x
	
class Transformer(lightning.LightningModule):

	def __init__(self,
			  n_amino:int,
			  amino_dim:int,
			  n_heads:int,
			  n_layers:int,
			  hidden_dim:int,
			  expansion_factor:int,
			  n_structures:int=2):
		
		super().__init__()
		
		self.embeddings = nn.Embedding(n_amino,amino_dim)
		layers = []

		for _ in range(n_layers):
			layers.append(EncoderBlock(hidden_dim,n_heads,expansion_factor))

		self.layers = nn.Sequential(*layers)
		self.masked_lm_head = nn.Linear(hidden_dim,n_amino)
		self.structure_head = nn.Linear(hidden_dim,n_structures)

	def forward(self,x,mask):

		x = self.embeddings(x)
		x = sin_pos_encode(x)

		for h in self.layers:
			x = h(x,mask)

		return x
	
	def lm_forward(self,x,mask):

		x = self(x,mask)
		x = self.masked_lm_head(x)
		return x
	
	def structure_forward(self,x,mask):

		x = self(x,mask)
		x = self.structure_head(x)
		return x
	
class HyenaOperator(lightning.LightningModule):

	def __init__(self,
			  hidden_dim:int,
			  kernel_size:int):
		
		super().__init__()
		
		self.k1 = nn.Conv1d(hidden_dim,hidden_dim,kernel_size,
					  groups=hidden_dim,padding=kernel_size//2)
		self.k2 = nn.Conv1d(hidden_dim,hidden_dim,kernel_size,
					  groups=hidden_dim,padding=kernel_size//2)
		
		self.proj  = nn.Linear(hidden_dim,3*hidden_dim)

	def forward(self,x,mask):

		proj = self.proj(x)
		proj = proj.to(torch.float32)
		q,k,v = einx.rearrange("b l (k d) -> k b d l",proj,k=3)
		y = F.relu(self.k1(q))
		y =  v*self.k2(k*y)
		y = einx.rearrange("b d l -> b l d",y)
		return y

class HyenaBlock(lightning.LightningModule):

	def __init__(self,
			  hidden_dim:int,
			  kernel_size:int):

		super().__init__()

		self.hyena_operator = HyenaOperator(hidden_dim,kernel_size)
		self.mlp = MLP(hidden_dim,[hidden_dim,hidden_dim],3)
		self.ln = nn.LayerNorm(hidden_dim)
		self.drop = nn.Dropout(0.1)

	def forward(self,x,mask):

		x = x + self.drop(self.ln(self.hyena_operator(x,mask))) + self.drop(self.ln(self.mlp(x)))
		return x
	
class HyenaFormer(lightning.LightningModule):

	def __init__(self,
			  n_amino:int,
			  amino_dim:int,
			  kernel_size:int,
			  n_layers:int,
			  n_structures:int=2):

		super().__init__()
		
		self.embeddings = nn.Embedding(n_amino,amino_dim)
		layers = []

		for _ in range(n_layers):
			layers.append(HyenaBlock(amino_dim,kernel_size))

		self.layers = nn.Sequential(*layers)
		self.masked_lm_head = nn.Linear(amino_dim,n_amino)
		self.structure_head = nn.Linear(amino_dim,n_structures)

	def forward(self,x,mask):

		x = self.embeddings(x)

		for h in self.layers:
			x = h(x,mask)

		return x
	
	def lm_forward(self,x,mask):

		x = self(x,mask)
		x = self.masked_lm_head(x)
		return x
	
	def structure_forward(self,x,mask):

		x = self(x,mask)
		x = self.structure_head(x)
		return x
	
class StrippedHyena(lightning.LightningModule):

	def __init__(self,
			  n_amino:int,
			  amino_dim:int,
			  kernel_size:int,
			  n_layers:int,
			  n_structures:int=2):

		super().__init__()
		
		self.embeddings = nn.Embedding(n_amino,amino_dim)
		layers = []

		for _ in range(n_layers//2):
			layers.append(HyenaBlock(amino_dim,kernel_size))
			layers.append(EncoderBlock(amino_dim,4,2))

		self.layers = nn.Sequential(*layers)
		self.masked_lm_head = nn.Linear(amino_dim,n_amino)
		self.structure_head = nn.Linear(amino_dim,n_structures)

	def forward(self,x,mask):

		x = self.embeddings(x)

		for h in self.layers:
			x = h(x,mask)

		return x
	
	def lm_forward(self,x,mask):

		x = self(x,mask)
		x = self.masked_lm_head(x)
		return x
	
	def structure_forward(self,x,mask):

		x = self(x,mask)
		x = self.structure_head(x)
		return x
	
class CNN(lightning.LightningModule):

	def __init__(self,
			n_amino:int,
			amino_dim:int,
			n_layers:int,
			kernel_size:int,
			n_structures:int=2):
		
		super().__init__()
		
		self.embeddings = nn.Embedding(n_amino,amino_dim)
		layers = []

		for _ in range(n_layers):
			layers.append(FactorizedConv(amino_dim,amino_dim,kernel_size))
			#layers.append(nn.Conv1d(amino_dim,amino_dim,kernel_size,padding=kernel_size//2))

		self.layers = nn.Sequential(*layers)
		self.masked_lm_head = nn.Linear(amino_dim,n_amino)
		self.structure_head = nn.Linear(amino_dim,2)
		self.drop = nn.Dropout(0.1)

	def forward(self,x,mask):

		x = self.embeddings(x)

		#x = einx.rearrange("b l d -> b d l",x)
		for h in self.layers:
			x = F.relu(self.drop(h(x)))

		#x = einx.rearrange("b d l -> b l d",x)

		return x
	
	def lm_forward(self,x,mask):

		x = self(x,mask)
		x = self.masked_lm_head(x)
		return x
	
	def structure_forward(self,x,mask):

		x = self(x,mask)
		x = self.structure_head(x)
		return x

class LSTM(lightning.LightningModule):

	def __init__(self,
			n_amino:int,
			amino_dim:int,
			n_layers:int,
			n_structures:int=2):
		
		super().__init__()
		
		self.embeddings = nn.Embedding(n_amino,amino_dim)
		self.layers = nn.LSTM(amino_dim,amino_dim,n_layers,batch_first=True,
						bidirectional=True)
		self.masked_lm_head = nn.Linear(amino_dim*2,n_amino)
		self.structure_head = nn.Linear(amino_dim*2,n_structures)

	def forward(self,x,mask):

		x = self.embeddings(x)
		out,(h,c) = self.layers(x)
		return out
	
	def lm_forward(self,x,mask):

		x = self(x,mask)
		x = self.masked_lm_head(x)
		return x
	
	def structure_forward(self,x,mask):

		x = self(x,mask)
		x = self.structure_head(x)
		return x
