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
from models import StrippedHyena,Transformer,HyenaFormer
import pandas as pd

TRAINING_PATH = "training_data/labels.txt"
DEVICE = "cuda"
torch.set_float32_matmul_precision('medium')

d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

amino_acids = list(d.values())

AMINO_TO_INT_DICT = {i:idx+3 for idx,i in enumerate(amino_acids)}
AMINO_TO_INT_DICT["<PAD>"] = 0
AMINO_TO_INT_DICT["<MASK>"] = 1
AMINO_TO_INT_DICT["<END>"] = 0
INT_TO_AMINO_DICT = {v:k for v,k in AMINO_TO_INT_DICT.items()}


SST2_TO_INT_DICT = {"<PAD>":0,"<END>":0,"H":1,"C":2,"E":2}
SST3_TO_INT_DICT = {"<PAD>":0,"<END>":0,"H":1,"C":2,"E":3}
SST8_TO_INT_DICT = {"<PAD>":0,"<END>":0,"H":1,"C":2,"E":3,"B":4,"G":5,"I":6,"T":7,"S":8}

INT_TO_SST2 = {v:k for v,k in SST2_TO_INT_DICT.items()}
INT_TO_SST3 = {v:k for v,k in SST3_TO_INT_DICT.items()}
INT_TO_SST8 = {v:k for v,k in SST8_TO_INT_DICT.items()}

def tokenize_from_dict(sequence,tokenizer_dict,length=None):

	if length is None:
		length = len(sequence)

	seq = []

	for i in range(length):

		if i>len(sequence):
			seq.append(tokenizer_dict["<PAD>"])
		elif i == len(sequence):
			seq.append(tokenizer_dict["<END>"])
		else:
			seq.append(tokenizer_dict[sequence[i]])

	return torch.tensor(seq)

def get_xy(df,l_max):

	inputs = []
	sst2_labels = []
	sst3_labels = []
	sst8_labels = []

	for index,row in df.iterrows():
		# we ignore sequences with special amino acids
		if "*" in row["seq"]:
			continue

		if len(row["seq"]) <= l_max-1:
			inputs.append(tokenize_from_dict(row["seq"],AMINO_TO_INT_DICT,l_max))
			sst2_labels.append(tokenize_from_dict(row["sst3"],SST2_TO_INT_DICT,l_max))
			sst3_labels.append(tokenize_from_dict(row["sst3"],SST3_TO_INT_DICT,l_max))
			sst8_labels.append(tokenize_from_dict(row["sst8"],SST8_TO_INT_DICT,l_max))

	return torch.stack(inputs),torch.stack(sst2_labels),torch.stack(sst3_labels),torch.stack(sst8_labels)

def one_hot(x, num_classes):
    return torch.eye(num_classes).to(x.device)[x]

def masked_accuracy(logits, labels, mask):
    predictions = torch.argmax(logits, axis=-1)
    correct_predictions = (predictions == labels) * mask
    accuracy = torch.sum(correct_predictions) / torch.sum(mask)
    return accuracy

def train_val_split(x,y,p):

	permutations = torch.randperm(len(x))
	p = int(len(x)*p)

	return x[permutations[:p]],y[permutations[:p]],x[permutations[p:]],y[permutations[p:]]

def get_batch_idx(n:int,
				  batch_size:int,
				  shuffle:bool=True):

	if shuffle:
		permutations = torch.randperm(n)
	else:
		permutations = torch.arange(n)
	
	permutations = permutations[0:batch_size*(n//batch_size)]
	batches = einx.rearrange("(n b) -> n b",permutations,b=batch_size)

	return batches

def mask_from_list(x,values):

	mask = torch.zeros_like(x,dtype=torch.bool)

	for i in values:
		mask += (x == i)

	return mask

def lm_mask(x,pad_mask,p=0.1):
	mask = torch.rand((pad_mask.shape))>(1-p)
	x[mask] = 1
	mask = torch.logical_and(mask,torch.logical_not(pad_mask))
	return x,mask

def structure_mask(x,pad_mask):
	return x,torch.logical_not(pad_mask)

def torch_train(fabric,
				model:Transformer,
				mask_function,
				x_train,
				y_train,
				x_test,
				y_test,
				batch_size,
				n_epochs,
				main_lr=1E-3,
				head_lr=1E-3):
	
	loss_fn = nn.CrossEntropyLoss(reduce=True,ignore_index=-100)
	
	def loss_acc(fabric,
			  model,
			  x_batch,
			  y_batch,
			  model_mask,
			  loss_mask):
			
		with fabric.autocast():

			x_batch = x_batch.to(DEVICE)
			y_batch = y_batch.to(DEVICE)
			model_mask = model_mask.to(DEVICE)
			loss_mask = loss_mask.to(DEVICE)
		
			y_pred = model.structure_forward(x_batch,model_mask)
			y_batch[torch.logical_not(loss_mask)] = -100
			loss = loss_fn(einx.rearrange("b l c -> b c l",y_pred),y_batch)
			acc = masked_accuracy(y_pred,y_batch,loss_mask)
		return loss,acc

	
	def val_loop(fabric,
			  model:Transformer,
			  x_test,
			  y_test,
			  batch_size,):
		
		
		batches = get_batch_idx(len(x_test),batch_size,False)

		val_losses = []
		val_acc = []

		model.eval()

		with torch.no_grad():

			for b,indices in enumerate(batches):

				x_batch = x_test[indices]
				y_batch = y_test[indices]

				pad_mask = mask_from_list(x_batch,[0,1])

				x_batch,masked = mask_function(x_batch,pad_mask)

				loss,acc = loss_acc(fabric,model,x_batch,y_batch,masked,masked)
				val_losses.append(loss.item())
				val_acc.append(acc.item())

		model.train()

		return np.mean(val_losses),np.mean(val_acc)
	
	optimizer = torch.optim.AdamW([
		{"params":model.layers.parameters(),
   		"lr":main_lr,
		"weight_decay":1E-2},
		{"params":model.masked_lm_head.parameters(),
   		"lr":head_lr,
		"weight_decay":1E-2},
		{"params":model.structure_head.parameters(),
   		"lr":head_lr,
		"weight_decay":1E-2},
	])
	
	model,optimizer = fabric.setup(model,optimizer)

	val_acc_results = []
	train_acc_results = []

	for epoch in range(n_epochs):

		batches = get_batch_idx(len(x_train),batch_size,True)

		losses = []
		accuracies = []

		for b,indices in enumerate(batches):

			x_batch = x_train[indices]
			y_batch = y_train[indices]

			# mask pad tokens

			pad_mask = mask_from_list(x_batch,[0,1])

			# mask (for language modeling)

			x_batch,masked = mask_function(x_batch,pad_mask)

			loss,acc = loss_acc(fabric,model,x_batch,y_batch,masked,masked)

			fabric.backward(loss)

			optimizer.step()
			optimizer.zero_grad()

			losses.append(loss.item())
			accuracies.append(acc.item())

		val_loss,val_acc = val_loop(fabric,model,x_test,y_test,batch_size,)

		train_acc_results.append(np.mean(accuracies))
		val_acc_results.append(val_acc)

		print(f"Training loss: {np.mean(losses)} | Training acc: {np.mean(accuracies)} | Val loss: {val_loss} | Vall acc: {val_acc}")

	return train_acc_results,val_acc_results

def lm_train(fabric,
				model:Transformer,
				mask_function,
				x_train,
				y_train,
				x_test,
				y_test,
				batch_size,
				n_epochs,
				main_lr=1E-3,
				head_lr=1E-3):
	
	loss_fn = nn.CrossEntropyLoss(reduce=True,ignore_index=-100)
	
	def loss_acc(fabric,
			  model:Transformer,
			  x_batch,
			  y_batch,
			  model_mask,
			  loss_mask):
			
		with fabric.autocast():

			x_batch = x_batch.to(DEVICE)
			y_batch = y_batch.to(DEVICE)
			model_mask = model_mask.to(DEVICE)
			loss_mask = loss_mask.to(DEVICE)
		
			y_pred = model.lm_forward(x_batch,model_mask)
			y_batch[torch.logical_not(loss_mask)] = -100
			loss = loss_fn(einx.rearrange("b l c -> b c l",y_pred),y_batch)
			acc = masked_accuracy(y_pred,y_batch,loss_mask)
		return loss,acc

	
	def val_loop(fabric,
			  model:Transformer,
			  x_test,
			  y_test,
			  batch_size,):
		
		
		batches = get_batch_idx(len(x_test),batch_size,False)

		val_losses = []
		val_acc = []

		with torch.no_grad():

			for b,indices in enumerate(batches):

				x_batch = x_test[indices]
				y_batch = y_test[indices]

				pad_mask = mask_from_list(x_batch,[0,1])

				x_batch,masked = mask_function(x_batch,pad_mask)

				loss,acc = loss_acc(fabric,model,x_batch,y_batch,masked,masked)
				val_losses.append(loss.item())
				val_acc.append(acc.item())

		return np.mean(val_losses),np.mean(val_acc)
	
	optimizer = torch.optim.AdamW([
		{"params":model.layers.parameters(),
   		"lr":main_lr,
		"weight_decay":1E-2},
		{"params":model.masked_lm_head.parameters(),
   		"lr":head_lr,
		"weight_decay":1E-2},
		{"params":model.structure_head.parameters(),
   		"lr":head_lr,
		"weight_decay":1E-2},
	])
	
	model,optimizer = fabric.setup(model,optimizer)

	val_acc_results = []
	train_acc_results = []

	for epoch in range(n_epochs):

		batches = get_batch_idx(len(x_train),batch_size,True)

		losses = []
		accuracies = []

		for b,indices in enumerate(batches):

			x_batch = x_train[indices]
			y_batch = y_train[indices]

			# mask pad tokens

			pad_mask = mask_from_list(x_batch,[0,1])

			# mask (for language modeling)

			x_batch,masked = mask_function(x_batch,pad_mask)

			loss,acc = loss_acc(fabric,model,x_batch,y_batch,masked,masked)

			fabric.backward(loss)

			optimizer.step()
			optimizer.zero_grad()

			losses.append(loss.item())
			accuracies.append(acc.item())

		val_loss,val_acc = val_loop(fabric,model,x_test,y_test,batch_size,)

		train_acc_results.append(np.mean(accuracies))
		val_acc_results.append(val_acc)

		print(f"Train perplexity: {math.exp(np.mean(losses))} | Training acc: {np.mean(accuracies)} | Val perplexity: {math.exp(val_loss)} | Vall acc: {val_acc}")

	return train_acc_results,val_acc_results



if __name__ == "__main__":

	df = pd.read_csv("extra_data/2018-06-06-pdb-intersect-pisces.csv")

	inputs,sst2_labels,sst3_labels,sst8_labels = get_xy(df,384)

	fabric = Fabric(accelerator="cuda",precision="bf16-mixed")

	fabric.launch()

	x_test,y_test,x_train,y_train = train_val_split(inputs,sst3_labels,0.05)

	hyena_model = StrippedHyena(len(AMINO_TO_INT_DICT),32,31,6,4)

	print(ModelSummary(hyena_model))
	
	print("Structure training")
	
	torch_train(fabric,
		model=hyena_model,
		mask_function=structure_mask,
		x_train=x_train,
		y_train=y_train,
		x_test=x_test, 
		y_test=y_test,
		batch_size=32,
		n_epochs=20,
		main_lr=1E-3,
		head_lr=1E-3)