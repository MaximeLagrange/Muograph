# Usual suspects
from typing import Dict, List, Union, Tuple, Optional
import torch
import functools
from functools import partial
import math
NoneType = type(None)
from fastprogress import progress_bar
from copy import deepcopy

# Muograph inports
from reconstruction.POCA import *
from tracking.Tracking import *
from volume.Volume_Interest import *


class neighborhood_sum():
    
    """
    Set up the neighborhood algorithmn, as developped by Vishal Kumar, GitHub: Vishal92Kumar.

    INPUT:
     - tracks:Tracking, an instance of the Tracking class.
     - output_dir:str, path to the directory where to save figures abd results.
     - cut_low_theta:0.001
    """

    def __init__(self, poca:POCA, output_dir:str):

        self.POCA = poca
        self.output_dir = output_dir + "NS/"
        self.create_directory(self.output_dir)

        
    def set_up(self, 
               epsilon:float=20, 
               dtheta_min_max:Tuple[float]=[.5*math.pi/180,math.pi], 
               save:bool=False,
               plot:bool=False) -> None:
        
        self.ns_params = {"epsilon":epsilon, 
                          "dtheta_min_max":dtheta_min_max}
        
        # get BCA name from its parameters
        self.ns_name = self.get_ns_name()
        
        # Remove low/high scattering events
        self.mask = (self.POCA.tracks.dtheta>dtheta_min_max[0]) & (self.POCA.tracks.dtheta<dtheta_min_max[1])
        
        # Get poca points position
        self.x, self.y, self.z = self.POCA.POCA_points[self.mask,0], self.POCA.POCA_points[self.mask,1], self.POCA.POCA_points[self.mask,2]
        
        # Get scattering angle
        dtheta = self.POCA.tracks.dtheta[self.mask]
        
        self.neighbor_sum = torch.zeros_like(dtheta)
        self.n_neighbor_3D = torch.zeros_like(dtheta)
        
        # Loop over events
        for event in progress_bar(range(len(dtheta))):
            
            distance_3D = torch.sqrt((self.x-self.x[event])**2+(self.y-self.y[event])**2+(self.z-self.z[event])**2)
            mask_3D_d = distance_3D<epsilon
            self.neighbor_sum[event] = dtheta[mask_3D_d].sum()
            self.n_neighbor_3D[event] = mask_3D_d.sum()
            
        # Compute normalized ns
        self.normalized_ns = self.neighbor_sum - torch.min(self.neighbor_sum) / (torch.max(self.neighbor_sum) - torch.min(self.neighbor_sum))
        
        # Save ns class instance
        if save:
            self.dir_name = self.output_dir + self.ns_name + "/"
            self.create_directory(dir_name=self.dir_name)
            self.save(self.dir_name)
            
        
    def create_directory(self, dir_name:str) -> None:
        
        import os
        if(os.path.exists(dir_name)==False):
            os.mkdir(dir_name)
            print("\n{} directory created".format(dir_name))

            
    def save(self,directory:str) -> None:
        
        import pickle
        
        with open(directory+"ns_class", "wb") as f:
            pickle.dump(self,f)
   

    def get_ns_name(self,) -> str:
        
        epsilon = "epsilon_{:.0f}_".format(self.ns_params["epsilon"])
        dtheta = "dtheta_min_max_{:.2f}_{:.2f}_rad".format(self.ns_params["dtheta_min_max"][0],self.ns_params["dtheta_min_max"][1])
        
        return epsilon+dtheta
        
        
    
    
    