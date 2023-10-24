# Usual suspects
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
import pandas as pd
from typing import List,Callable

# Librairy
from tracking.Tracking import *
from volume.Volume_Interest import *
from reconstruction.POCA import mean_list, median_list, RMS_list

class AbsorbtionReconstruction():

    def __init__(self, tracks:Tracking=None, from_save:str=None):

        r"""
        Class for absorption image reconstruction based on ASR algorithm.
        """

        if((tracks is not None)):
            self.tracks = tracks
            self.voi = tracks.voi

            # find sub volume
            self.sub_vol_indices = self.find_sub_volume()

            # find triggered voxels
            self.triggered_voxels = self.find_triggered_voxels()

            # compute path within VOI
            self.path = self.compute_path_within_voi()

        if(from_save is not None):
            with open(from_save,'rb') as f:
                self.triggered_voxels = pickle.load(f)

    def find_sub_volume(self)->List[Tensor]:
        r"""
        For each muon event, find the part of the VOI where voxels will be triggered
        by a muon track. When searching for triggered voxel for each track, it is better 
        to search within this sub-volume rather than within the whole VOI.
        This step is just here to help reduce computing time.
        The sub-volume will be then used as an input by 
        the find_triggered_voxel() method.

        IUNPUT: 
        - tracks.xyz_in_VOI: tensor with size [3,2,n_event] Muon position when entering/exiting the VOI
        for the INCOMING muon track
        - voi.voxel_edges: tensor with size [n_vox_xyz[0],n_vox_xyz[1],n_vox_xyz[2],2,3] Voxels edges position

        OUTPUT: 
        sub_vol_indices: List containing the indices of potentially triggered voxels. len() = n_event
        """
        sub_vol_indices = []

        print("\nSub-volumes")
        for event in progress_bar(range(self.tracks.n_event)):          

            x_min = torch.min(self.tracks.xyz_in_VOI[0,:,event])
            x_max = torch.max(self.tracks.xyz_in_VOI[0,:,event])
            
            y_min = torch.min(self.tracks.xyz_in_VOI[1,:,event])
            y_max = torch.max(self.tracks.xyz_in_VOI[1,:,event])
            
            sub_vol_indices.append(((self.voi.voxel_edges[:,:,:,1,0]>x_min) & 
                                    (self.voi.voxel_edges[:,:,:,1,1]>y_min) & 
                                    (self.voi.voxel_edges[:,:,:,0,0]<x_max) &
                                    (self.voi.voxel_edges[:,:,:,0,1]<y_max)).nonzero())
        print("DONE")
        return sub_vol_indices
    
    def find_triggered_voxels(self)->List[Tensor]:

        r"""
        For each muon outgoing tracks, find the associated triggered voxels.

        OUTPUT:
        triggererd_voxels: List with len() = n_event, containing the indices of triggered voxels as a Tensor
        (with size [n_triggered_vox,3])
        """
        triggererd_voxels = []

        print("\nVoxel triggering")
        for event in progress_bar(range(self.tracks.n_event)):
            mask = self.sub_vol_indices[event]
            if(len(mask)!=0):
                ix_min,iy_min,iz_min = torch.min(mask[:,0]).item(),torch.min(mask[:,1]).item(),torch.min(mask[:,2]).item()
                ix_max,iy_max,iz_max = torch.max(mask[:,0]).item(),torch.max(mask[:,1]).item(),torch.max(mask[:,2]).item()

                sub_voi_edges = self.voi.voxel_edges[ix_min:ix_max+1,iy_min:iy_max+1,iz_min:iz_max+1]
                sub_voi_edges = sub_voi_edges[:,:,:,:,None,:]
                sub_voi_edges = sub_voi_edges.expand(-1,-1,-1,-1,len(self.tracks.xyz_discrete_in[0,:,event]),-1)

                sub_mask_out = ((sub_voi_edges[:,:,:,0,:,0] < self.tracks.xyz_discrete_in[0,:,event]) & 
                                (sub_voi_edges[:,:,:,1,:,0] > self.tracks.xyz_discrete_in[0,:,event]) &
                                (sub_voi_edges[:,:,:,0,:,1] < self.tracks.xyz_discrete_in[1,:,event]) & 
                                (sub_voi_edges[:,:,:,1,:,1] > self.tracks.xyz_discrete_in[1,:,event]) &
                                (sub_voi_edges[:,:,:,0,:,2] < self.tracks.xyz_discrete_in[2,:,event]) & 
                                (sub_voi_edges[:,:,:,1,:,2] > self.tracks.xyz_discrete_in[2,:,event]))

                mask = sub_mask_out
                vox_list = mask.nonzero()[:,:-1].unique(dim=0)
                vox_list[:,0]+=ix_min
                vox_list[:,1]+=iy_min
                vox_list[:,2]+=iz_min
                triggererd_voxels.append(vox_list)
            else:
                triggererd_voxels.append([])

        print("DONE")
        return triggererd_voxels
    
    def compute_path_within_voi(self) -> Tensor:

        r"""
        Copmutes the muon path within the VOI based on the muon outgoing tracks

        """

        dx = torch.abs(self.tracks.xyz_in_VOI[0,0]-self.tracks.xyz_in_VOI[0,1])
        dy = torch.abs(self.tracks.xyz_in_VOI[1,0]-self.tracks.xyz_in_VOI[1,1])
        dz = torch.abs(self.tracks.xyz_in_VOI[2,0]-self.tracks.xyz_in_VOI[2,1])

        return torch.sqrt(dx**2+dy**2+dz**2)

    def generate_score_list(self) -> List[List[List[List]]]:
    # Generate scores
        score_list = torch.zeros((self.voi.n_vox_xyz[0],
                                self.voi.n_vox_xyz[1],
                                self.voi.n_vox_xyz[2])).tolist()

        for i in range(self.voi.n_vox_xyz[0]):
            for j in range(self.voi.n_vox_xyz[1]):
                for k in range(self.voi.n_vox_xyz[2]):
                    score_list[i][j][k]=[]
                
        return score_list

    def assign_score_voxel(self,
                           score_list:List[List[List[List]]],
                           score_feature:Tensor, 
                           masks:Union[List[Tensor],Tensor] = None) -> Tuple[List[List[List[List]]],int]:

        full_mask=0

        if masks is None:
            full_mask = torch.ones_like(self.tracks.theta,dtype=bool)
        
        elif(type(masks)==list):
            full_mask = torch.ones_like(masks[0],dtype=bool)
            for mask in masks:
                full_mask = full_mask & mask
        else:
            full_mask = masks
        
        n_true = 0
        print("assigning voxel's score")
        for i,vox_list in enumerate(progress_bar(self.triggered_voxels)):
            if(full_mask[i]):
                n_true+=1
            for vox in vox_list: 
                ix,iy,iz=vox[0],vox[1],vox[2]
                if(full_mask[i]):
                    score_list[ix][iy][iz].append(score_feature[i].item())
                else:
                    score_list[ix][iy][iz].append(0)
                    
        return (score_list, n_true)


    def compute_final_voxel_score(self,
                                 score_list:List[List[List[List]]],
                                 score_method:Callable,
                                 quartile:Optional[int]=None) -> Tensor:
        
        final_voxel_scores = torch.zeros(self.voi.n_vox_xyz[0],
                                            self.voi.n_vox_xyz[1],
                                            self.voi.n_vox_xyz[2])
        
        print("computing final voxel's score")
        for i in progress_bar(range(self.voi.n_vox_xyz[0])):
            for j in range(self.voi.n_vox_xyz[1]):
                for k in range(self.voi.n_vox_xyz[2]):

                    if(score_list[i][j][k]!=[]):
                        if(quartile is not None):
                            final_voxel_scores[i,j,k] = score_method(score_list[i][j][k],quartile=quartile)
                        else:
                            final_voxel_scores[i,j,k] = score_method(score_list[i][j][k])
                    else:
                        final_voxel_scores[i,j,k] = 0.
                        
        return final_voxel_scores
    
    def save_triggered_voxels(self,filename:str):

        with open(filename, 'wb') as f:
            pickle.dump(self.triggered_voxels,f)

    def save(self,filename:str):

        with open(filename,'wb') as f:
            pickle.dump(self,f)
