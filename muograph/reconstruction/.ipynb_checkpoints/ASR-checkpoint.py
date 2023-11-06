# Usual suspects
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
import pandas as pd
from typing import List,Callable
import sys
import functools
from functools import partial
NoneType = type(None)
# Librairy
from tracking.Tracking import *
from volume.Volume_Interest import *

class ASR():

    def __init__(self, output_dir:str, tracks:Tracking=None, triggered_voxel_file:str=None):
        
        """
        Class used for Angle Statistic Reconstruction algorithm, from  
        "Angle Statistics Reconstruction: a robust reconstruction algorithm for Muon Scattering Tomography", 
        2014 JINST 9 P11019, (http://iopscience.iop.org/1748-0221/9/11/P11019).
        
        INPUT:
         - output_dir:str, path to the directory where to save figures and variables.
         - tracks:Tracking, and instance of the Tracking class.
         - triggered_voxel_file:str, path to the file containing the triggered voxels. If provided, the
         list of triggered voxels will be loaded from this file and will not be computed.
        """
        
        assert (triggered_voxel_file is not None)|(tracks is not None), "Provide at least tracks or triggered_voxel_file path!"
        
        # Create output directory
        self.output_dir = output_dir+"ASR/"
        self.create_directory(self.output_dir)
        
        if(triggered_voxel_file is None):
            self.tracks = tracks
            self.voi = tracks.voi

            # find sub volume
            self.sub_vol_indices = self.find_sub_volume()
            
            # find triggered voxels
            self.triggered_voxels = self.find_triggered_voxels()
            
            self.save_triggered_voxels(self.output_dir)
            
        elif(triggered_voxel_file is not None):
            with open(triggered_voxel_file,'rb') as f:
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
        - tracks.xyz_out_VOI: tensor with size [3,2,n_event] Muon position when entering/exiting the VOI
        for the OUTGOING muon track
        - voi.voxel_edges: tensor with size [n_vox_xyz[0],n_vox_xyz[1],n_vox_xyz[2],2,3] Voxels edges position

        OUTPUT: 
        sub_vol_indices: List containing the indices of potentially triggered voxels. len() = n_event
        """
        sub_vol_indices = []

        print("\nSub-volumes")
        for event in progress_bar(range(self.tracks.n_event)):
            
            x_min = torch.min(torch.min(self.tracks.xyz_in_VOI[0,:,event]),torch.min(self.tracks.xyz_out_VOI[0,:,event]))
            x_max = torch.max(torch.max(self.tracks.xyz_in_VOI[0,:,event]),torch.max(self.tracks.xyz_out_VOI[0,:,event]))
            
            y_min = torch.min(torch.min(self.tracks.xyz_in_VOI[1,:,event]),torch.min(self.tracks.xyz_out_VOI[1,:,event]))
            y_max = torch.max(torch.max(self.tracks.xyz_in_VOI[1,:,event]),torch.max(self.tracks.xyz_out_VOI[1,:,event]))
            
            sub_vol_indices.append(((self.voi.voxel_edges[:,:,:,1,0]>x_min) & 
                                    (self.voi.voxel_edges[:,:,:,1,1]>y_min) & 
                                    (self.voi.voxel_edges[:,:,:,0,0]<x_max) &
                                    (self.voi.voxel_edges[:,:,:,0,1]<y_max)).nonzero())
        print("DONE")
        return sub_vol_indices
    
    
    def find_triggered_voxels(self)->List[Tensor]:

        r"""
        For each muon incoming and outgoing tracks, find the associated triggered voxels.
        Only voxels triggered by both INCOMING and OUTGOING tracks are kept.

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
                sub_voi_edges = sub_voi_edges.expand(-1,-1,-1,-1,len(self.tracks.xyz_discrete_out[0,:,event]),-1)

                sub_mask_in = ((sub_voi_edges[:,:,:,0,:,0] < self.tracks.xyz_discrete_in[0,:,event]) & 
                                (sub_voi_edges[:,:,:,1,:,0] > self.tracks.xyz_discrete_in[0,:,event]) &
                                (sub_voi_edges[:,:,:,0,:,1] < self.tracks.xyz_discrete_in[1,:,event]) & 
                                (sub_voi_edges[:,:,:,1,:,1] > self.tracks.xyz_discrete_in[1,:,event]) &
                                (sub_voi_edges[:,:,:,0,:,2] < self.tracks.xyz_discrete_in[2,:,event]) & 
                                (sub_voi_edges[:,:,:,1,:,2] > self.tracks.xyz_discrete_in[2,:,event]))

                sub_mask_out = ((sub_voi_edges[:,:,:,0,:,0] < self.tracks.xyz_discrete_out[0,:,event]) & 
                                (sub_voi_edges[:,:,:,1,:,0] > self.tracks.xyz_discrete_out[0,:,event]) &
                                (sub_voi_edges[:,:,:,0,:,1] < self.tracks.xyz_discrete_out[1,:,event]) & 
                                (sub_voi_edges[:,:,:,1,:,1] > self.tracks.xyz_discrete_out[1,:,event]) &
                                (sub_voi_edges[:,:,:,0,:,2] < self.tracks.xyz_discrete_out[2,:,event]) & 
                                (sub_voi_edges[:,:,:,1,:,2] > self.tracks.xyz_discrete_out[2,:,event]))

                mask = sub_mask_in & sub_mask_out
                vox_list = mask.nonzero()[:,:-1].unique(dim=0)
                vox_list[:,0]+=ix_min
                vox_list[:,1]+=iy_min
                vox_list[:,2]+=iz_min
                triggererd_voxels.append(vox_list)
            else:
                triggererd_voxels.append([])

        print("DONE")
        return triggererd_voxels
    
    
    def generate_score_list(self,VOI:VolumeInterest) -> List[List[List[List]]]:
        
        """
        Generates a nested List with size (Nx,Ny,Nz,0), where Nx,Ny,Nz are the number of voxels
        along the x,y,z direction.
        
        OUTPUT:
         - score_list:List[List[List[List]] with size (Nx,Ny,Nz,0).
        """
    # Generate scores
        score_list = torch.zeros((VOI.n_vox_xyz[0],
                                 VOI.n_vox_xyz[1],
                                 VOI.n_vox_xyz[2])).tolist()

        for i in range(VOI.n_vox_xyz[0]):
            for j in range(VOI.n_vox_xyz[1]):
                for k in range(VOI.n_vox_xyz[2]):
                    score_list[i][j][k]=[]
                
        return score_list

    
    def assign_score_voxel(self,
                           triggered_voxels:List[Tensor],
                           score_list:List[List[List[List]]],
                           score_feature:Tensor,
                           masks:Union[List[Tensor],Tensor] = None) -> Tuple[List[List[List[List]]],int]:
        
        """
        Append scores to the list of each voxels. Voxel's list will then be used to compute
        voxel's final score.
        
        INPUT:
         - triggered_voxels:List[torch.tensor] with length (Nevent). Each element of the 
         list is a tensor with size Nhit, the number of muons hitting (passing through)
         the voxel.
         - score_list:List[List[List[List]] with size (Nx,Ny,Nz,0) which stores voxels scores.
         - score_feature:Tensor with size (Nevent), the score appended to voxels list.
         - masks:Union[List[Tensor],Tensor], a tensor or list of tensor. Events for which 
         mask==False will be discarded. Set to None by default.
         
         OUTPUT:
          - score_list:List[List[List[List]] with size (Nx,Ny,Nz,0) which stores voxels scores.
          - n_true
        """

        full_mask=0

        if masks is None:
            full_mask = torch.ones_like(len(triggered_voxels),dtype=bool)
        
        elif(type(masks)==list):
            full_mask = torch.ones_like(len(triggered_voxels),dtype=bool)
            for mask in masks:
                full_mask = full_mask & mask
        else:
            full_mask = masks
        
        n_true = 0
        print("assigning voxel's score")
        for i,vox_list in enumerate(progress_bar(triggered_voxels)):
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
                                  VOI:VolumeInterest,
                                  score_list:List[List[List[List]]],
                                  score_method:functools.partial=partial(np.quantile,q=.5),
                                  quartile:Optional[int]=None) -> Tensor:
        
        final_voxel_scores = torch.zeros(VOI.n_vox_xyz[0],
                                         VOI.n_vox_xyz[1],
                                         VOI.n_vox_xyz[2])
        
        print("computing final voxel's score")
        for i in progress_bar(range(VOI.n_vox_xyz[0])):
            for j in range(VOI.n_vox_xyz[1]):
                for k in range(VOI.n_vox_xyz[2]):

                    if(score_list[i][j][k]!=[]):
                        if(quartile is not None):
                            final_voxel_scores[i,j,k] = score_method(score_list[i][j][k])
                        else:
                            final_voxel_scores[i,j,k] = score_method(score_list[i][j][k])
                    else:
                        final_voxel_scores[i,j,k] = 0.
                        
        return final_voxel_scores
    
    
    def save_triggered_voxels(self,directory:str):

        with open(directory+"asr_triggered_voxels", 'wb') as f:
            pickle.dump(self.triggered_voxels,f)
            
            
    def save(self,directory:str,triggered_vox_only:bool=False) -> None:
        if(triggered_vox_only):
            self.tracks=None
            self.sub_vol_indices = None
            
            with open(directory+"ASR_class", 'wb') as f:
                pickle.dump(self,f)
        else:
            with open(directory+"ASR_class", 'wb') as f:
                pickle.dump(self,f)

                
    def asr_analysis(self,
                     VOI:VolumeInterest,
                     dtheta:torch.tensor, 
                     dtheta_range:List[float]=[0.,math.pi/2],
                     score_method:functools.partial=partial(np.quantile,q=.5),
                     save:bool=True,
                     plot:bool=True) -> torch.tensor:
        
        def get_partial_name_args(func:Union[functools.partial,NoneType]) -> Union[str]:

            if(func is not None):
                func_name = func.func.__name__
                args, values = list(func.keywords.keys()), list(func.keywords.values())

                for i, arg in enumerate(args):
                    func_name+="_{}={}".format(arg,values[i])
                    
                return func_name
            else:
                return "None"
        
        # get params
        self.asr_params = {"score_method":get_partial_name_args(score_method),
                           "dtheta_range":dtheta_range}
        
        # get asr analysis name
        self.asr_name = self.get_asr_name()
        
        # create output directory with BCA name
        self.dir_name = self.output_dir + self.asr_name + "/"
        self.create_directory(dir_name=self.dir_name)
        
        # ANALYSIS
        # create empty score list
        score_list = self.generate_score_list(VOI)
    
        # Define mask to filter events
        mask_dtheta = (dtheta>dtheta_range[0])&(dtheta<dtheta_range[1])

        # Assign score if the event respects the conditions in mask
        score_list,n_true = self.assign_score_voxel(triggered_voxels=self.triggered_voxels,
                                                    score_list=score_list,
                                                    score_feature=dtheta,
                                                    masks = mask_dtheta)

        # Compute final voxel scores using the desired function
        final_scores = self.compute_final_voxel_score(VOI=VOI,
                                                     score_list=score_list,
                                                     score_method=score_method)
        
        if(save):
            with open(self.dir_name+"asr_final_scores", "wb") as f:
                pickle.dump(final_scores,f)
                
        if(plot):
            from plotting.Plotting import plot_VOI_pred
            plot_VOI_pred(preds=final_scores, filename=self.dir_name+"asr_plot")
            
        
        return final_scores
    
    
    def get_asr_name(self,) -> str:
        
        method = "method_{}_".format(self.asr_params["score_method"])
        dtheta = "{:.2f}_{:.2f}_rad_".format(self.asr_params["dtheta_range"][0],self.asr_params["dtheta_range"][1])
        
        asr_name = method+dtheta
        
        return asr_name
    
    
    def create_directory(self, dir_name:str) -> None:
        
        import os
        if(os.path.exists(dir_name)==False):
            os.mkdir(dir_name)
            print("\n{} directory created".format(dir_name))