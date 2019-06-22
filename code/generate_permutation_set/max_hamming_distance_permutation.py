#DOWNLOAD THE DATA
from google.colab import drive
import zipfile

import torch
from sympy.utilities.iterables import multiset_permutations
import numpy as np
import pandas as pd

#MOUNT GDRIVE: DOWNLOAD DATA AND FILTERED LABELS
drive.mount('/content/gdrive',force_remount=True)
print("mounted google drive")

## UNZIP ZIP
#print ("Uncompressing zip file")
#zip_ref = zipfile.ZipFile('/content/gdrive/My Drive/CheXpert-v1.0-small.zip', 'r')
#zip_ref.extractall()
#zip_ref.close()
#print("downloaded files")

file_name_p_set = "checkpoint_p_set.pt"
file_name_P_= "checkpoint_P_.pt"
file_name_i = "checkpoint_i.txt"

PATH_p_set = F"gdrive/My Drive/summerthesis/permutation_set/{file_name_p_set}" 
PATH_P_ = F"gdrive/My Drive/summerthesis/permutation_set/{file_name_P_}" 
PATH_i = F"gdrive/My Drive/summerthesis/permutation_set/{file_name_i}" 


use_cuda = True
if use_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    print("CUDA didn't work")
    device = torch.device('cpu')

def calculate_avg_hamming(pset):
  cardinality=p_set.shape[0]
    
  D = torch.Tensor().to(device=device)
  
  sum_=0
  for idx,p in enumerate(pset):
    P_= torch.cat((pset[:idx,:],pset[idx+1:,:]))
    
    #hamming distances of ith for all other P'
    D_i = torch.stack([(p != p_prime) for p_prime in P_],dim=0).sum(dim=0).view(1,-1).type(torch.FloatTensor).to(device=device)
    avg_i = D_i.sum(dim=1)/ (cardinality-1)
    sum_=avg_i+sum_
    
  
  return sum_ /cardinality


def pick_j(p_set, P_):
  
  D = torch.Tensor().to(device=device)
  for p in p_set:
    #hamming distances of ith for all other P'
    D_i = torch.stack([(p != p_prime) for p_prime in P_[1:,:].t()],dim=1).sum(dim=0).view(1,-1).type(torch.FloatTensor).to(device=device)
    D=torch.cat((D,D_i)).to(device = device)
  
  D = D.sum(dim=0).view(1,-1).to(device=device)
  j = P_[0,torch.argmax(D).item()].item()#first row of P_ is index checking which indexed perm is max

  return int(j)


set_p = np.array([0,1,2,3,4,5,6,7,8])
P_ = torch.stack([ torch.FloatTensor(p) for p in multiset_permutations(set_p)],dim=1).to(device=device)
_,perm_size = P_.shape

idx = torch.arange(perm_size).type(torch.FloatTensor).view(1,-1).to(device=device)
P_=torch.cat((idx,P_))

p_set = torch.Tensor().to(device=device)
j=np.random.choice(perm_size)

i = 0
checkpoint = 1
cardinality = 110

while i <= cardinality :
  
  if checkpoint == 0:
    
    indice = (P_[0,:] == j).nonzero()
    p_j = P_[1:,indice].view(1,-1).to(device=device)
    p_set = torch.cat((p_set,p_j)).to(device=device)
    P_ = torch.cat((P_[:,:indice] , P_[:,indice+1:]),dim=1).to(device=device)
    
  else:
    p_set = torch.load(PATH_p_set)
    P_ = torch.load(PATH_P_)#index is the first row
    with open(PATH_i, "r") as text_file:
      i =int(text_file.read())
      
    checkpoint=0
    print("starting from checkpoint i=",i)
    
  if i % 10 == 0:
    print("i:",i)
 
  if i == cardinality :
  
    torch.save(p_set, PATH_p_set)
    torch.save(P_, PATH_P_)

    with open(PATH_i, "w") as text_file:
      text_file.write(str(p_set.shape[0]))
      
    print('saved perm set to google drive')
  
  j = pick_j(p_set, P_)
  i = i+1


print("now checking uniqueness")
output = torch.unique(p_set,dim=0)
print("p_set",p_set.shape)
print("output(uniqie row)",output.shape)
print(p_set)

##HOOOOOOOOP


print("p_set size x N")
print(p_set.shape)
calculate_avg_hamming(p_set)
