# Usual suspects
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import pandas as pd
from typing import Union, List, Tuple

def hits_csv_to_tensor(filename:Union[str,List[str]] = '../data/Wooden_statue_copper_cylinder/DataCu.csv',
                       n_event:int = None) -> Tuple[Tensor,pd.Series]:
    if(type(filename)==str):
        df = pd.read_csv(filename)
    if(type(filename)==list):
        dfs = [pd.read_csv(file) for file in filename]
        df = pd.concat(dfs,axis="rows")
        
    n_plane = len([ col for col in df.columns if "X" in col])
    hits = torch.zeros((3,n_plane,len(df)))

    for plane in range(n_plane):
        hits[0,plane,:]=torch.tensor(df['X'+str(plane)].to_list())
        hits[1,plane,:]=torch.tensor(df['Y'+str(plane)].to_list())
        hits[2,plane,:]=-torch.tensor(df['Z'+str(plane)].to_list())

    E = torch.tensor(df.E.values)
    if(n_event is not None):
        hits = hits[:,:,:n_event] 
        E = E[:n_event]
    return (hits, E)