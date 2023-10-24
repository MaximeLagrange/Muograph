import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from typing import Dict, List, Union, Tuple, Optional
# from __future__ import annotations


# Display latex
from IPython.display import display, Math


def norm(u,proj:str=None):
    if(proj is None):
        _sum=torch.sum(u**2)
        return torch.sqrt(_sum)

    if(proj=='xz'):
        return torch.sqrt(u[0]**2+u[2]**2)
    if(proj=='yz'):
        return torch.sqrt(u[1]**2+u[2]**2)
    if(proj=='xy'):
        return torch.sqrt(u[0]**2+u[1]**2)

def dot_product(u,v,proj:str=None):
    if(proj is None):
        return torch.sum(u*v)
    if(proj=='xz'):
        return u[0]*v[0]+u[2]*v[2]
    if(proj=='yz'):
        return u[1]*v[1]+u[2]*v[2]
    if(proj=='xy'):
        return u[1]*v[1]+u[0]*v[0]

def project(u,proj:str):
    '''
    - 'xy' = projection in xy plane
    - 'yz' = projection in yz plane
    - 'xz' = projection in xy plane
    '''
    if(proj=='xy'):
        return torch.array([u[0],u[1]])
    if(proj=='xz'):
        return torch.array([u[0],u[2]])
    if(proj=='yz'):
        return torch.array([u[1],u[2]])

def angle_u_v(u,v,proj:str=None):
    if(proj is None):
        return torch.arccos(dot_product(u,v)/(norm(u)*norm(v)))
    else:
        return torch.arccos(dot_product(u,v,proj=proj)/(norm(u,proj=proj)*norm(v,proj=proj)))

def vector_from_norm_direction(_norm:float,_theta_x:float,_theta_y:float,z_neg:bool=True):
    
    dz = _norm/(torch.sqrt(1+torch.tan(_theta_x)**2+torch.tan(_theta_y)**2))
    dx,dy = (dz*torch.tan(_theta_x),dz*torch.tan(_theta_y))
    if(z_neg):
        return torch.tensor([dx,dy,-dz])
    else:
        return torch.tensor([dx,dy,dz])