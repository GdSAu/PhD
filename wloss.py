import torch
from torch import linalg as LA
import numpy as np

def geodesic_error(M,M_prima):
    '''ground_truth, prediction'''
    M_biprimo = M*torch.linalg.inv(M_prima)
    g_error = torch.acos((M_biprimo[:,0,0]+M_biprimo[:,1,1]+M_biprimo[:,2,2]-1)/2)
    return g_error 

def weighted_loss(output, ground_truth, alpha = 0.5):
   '''a-> Traslation error
    b-> Rotation error'''
   alpha = alpha
   max = 0.4 # 0.396598
   
   #Divide y arregla dimensiones
   l =len(output)
   s = ground_truth[:,:3]
   r = ground_truth[:,3:].reshape([l,3,3])
   s_1 = output[:,:3]
   r_1 = output[:,3:].reshape([l,3,3])
   
   a = LA.norm((s-s_1), dim=-1, ord=2) / max
   b = geodesic_error((r*180)/np.pi,(r_1*180)/np.pi)
   loss = alpha*a + (0-alpha)*b
   return torch.nanmean(loss),a,b