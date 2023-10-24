# Usual suspects
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
import pandas as pd
import functools
from functools import partial
NoneType = type(None)

from typing import Dict, List, Union, Tuple, Optional
# from __future__ import annotations
import copy

# Display latex
from IPython.display import display, Math

from tracking.Tracking import *
from volume.Volume_Interest import *

def save_POCA(final_voxel_scores:Tensor,POCA_points:Tensor,filename:str)->None:
        
        torch.save(final_voxel_scores, filename+'voxel_scores.pt')
        torch.save(POCA_points, filename+'POCA_points.pt')

def Normalise_density_pred(density_pred:Tensor):
    
    return (density_pred-torch.min(density_pred))/(torch.max(density_pred)-torch.min(density_pred))



def mean_list(vox_list:List)->float:
    '''
    return the mean of the input list
    '''
    return np.mean(vox_list)

def median_list(vox_list:List, quartile:int)->float:
    '''
    Returns the quartile of the input list.
    Quartile:int must respect 0 < quartile < 100
    '''
    return np.percentile(vox_list,quartile)

def RMS_list(vox_list:List)->float:
    '''
    Return the RMS of the input list.
    '''
    return np.sqrt(np.sum(np.array(vox_list)**2))

def len_list(vox_list:List)->int:
    '''
    return the length of the list
    '''
    return len(vox_list)

class POCA:

    
    def __init__(self, output_dir:str, tracks:Tracking, cut_low_theta:float=0.0001)->None:
        
         # Create output directory
        self.output_dir = output_dir+"POCA/"
        self.create_directory(self.output_dir)
        
        self.tracks = copy.deepcopy(tracks)
        print('\nTotal # event = ', len(tracks.dtheta))

        # Compute a mask rejecting events with low scattering angles 
        # (theta<0.01 rad = 0.5 deg)
        self.mask_low_dtheta = tracks.dtheta>cut_low_theta # in rad

        # Discard events with dtheta<0.01 rad
        self.tracks.Apply_mask(self.mask_low_dtheta)
        print('# event after removing low theta events = ', len(self.tracks.dtheta))

        # Volume of interest
        self.VOI = self.tracks.voi

        # POCA points
        self.POCA_points = self.compute_POCA_points()

        # Discard events for which POCA point is outside the VOI
        self.mask_inside_VOI = self.compute_mask_VOI()

        # Filter POCA points
        self.POCA_points = self.POCA_points[self.mask_inside_VOI]

        # Filter events with POCA points outside VOI
        self.tracks.Apply_mask(self.mask_inside_VOI)

        # Mask associated with events filtering
        self.total_mask = self.compute_total_mask()

        # Voxel indices associated with POCA point location
        self.indices = self.assign_voxel_POCA()

        
    def create_directory(self, dir_name:str) -> None:

            import os
            if(os.path.exists(dir_name)==False):
                os.mkdir(dir_name)
                print("\n{} directory created".format(dir_name))
        
        
    def compute_total_mask(self) -> Tensor:

        total_mask = torch.zeros_like(self.mask_low_dtheta,dtype=bool)
        for i in progress_bar(range(len(self.mask_inside_VOI))):
            if(self.mask_inside_VOI[i]):
                total_mask[np.flatnonzero(self.mask_low_dtheta)[i]]=True

        return total_mask

    
    def compute_POCA_points(self)->Tensor:

        from numpy import cross

        '''
        @MISC {3334866,
        TITLE = {Closest points between two lines},
        AUTHOR = {Brian (https://math.stackexchange.com/users/72614/brian)},
        HOWPUBLISHED = {Mathematics Stack Exchange},
        NOTE = {URL:https://math.stackexchange.com/q/3334866 (version: 2019-08-26)},
        EPRINT = {https://math.stackexchange.com/q/3334866},
        URL = {https://math.stackexchange.com/q/3334866}
        }
        
        Compute POCA points. 
        INPUT: incoming and outgoing reconstructed tracks (Tensor[Nevent][3], Tensor[Nevent][3])
        OUTPUT: POCA points (Tensor[Nevent][3])
        
        Given 2 lines L1, L2 aka incoming and outgoing tracks with parametric equation:
        L1 = P1 + t*V1
        
        1- A segment of shortest length between two 3D lines L1 L2 is perpendicular to both lines (if L1 L2 are neither parallele or in the same plane). One must compute V3, vector perpendicular to L1 and L2
        2- Search for points where L3 = P1 + t1*V1 +t3*V3 crosses L2. One must find t1 and t2 for which:
        L3 = P1 + t1*V1 +t3*V3 = P2 + t2*V2
        
        3- Then POCA location M is the middle of the segment Q1-Q2 where Q1,2 = P1,2 +t1,2*V1,2
        '''
        #need to remove event with super small scattering angle
        # == cannot find closest point between two lines if they are parallel!!!

        P1, P2 = self.tracks.point_up[:], self.tracks.point_down[:]
        V1, V2 = self.tracks.tracks_up[:], self.tracks.tracks_down[:]
        
        V3 = torch.tensor(cross(V2,V1))

        RES = P2 - P1
        LES = torch.transpose(torch.stack([V1,-V2,V3]),0,1)
        LES = torch.transpose(LES,-1,1)

        ts = torch.linalg.solve(LES,RES)

        t1 = torch.stack([ts[:,0],ts[:,0],ts[:,0]],-1)
        t2 = torch.stack([ts[:,1],ts[:,1],ts[:,1]],-1)

        Q1s,Q2s = P1+t1*V1, P2+t2*V2
        M = (Q2s-Q1s)/2+Q1s

        return M

    
    def assign_voxel_POCA(self)->list:

        '''
        - Assign a voxel to a POCA point for each event
        
        - INPUT: self.POCA_points, the POCA locations
        - STATIC: Voxel edges positions
        - OUTPUT: INDICES, list of 1D tensor with size 3, containing the X,Y and Z index of the hit voxel
        len(INDICES) = len(self.events[self.POCA_mask])
        
        '''
        from fastprogress import progress_bar
        
        indices = torch.ones((len(self.POCA_points),3))-2
        
        print('\nAssigning voxel to each POCA point:')

        for i in progress_bar(range(len(self.POCA_points))):

            mask_vox_x1 = (self.POCA_points[i,0]>=self.VOI.voxel_edges[:,:,:,0,0])
            mask_vox_x2 = (self.POCA_points[i,0]<=self.VOI.voxel_edges[:,:,:,1,0])

            mask_vox_y1 = (self.POCA_points[i,1]>=self.VOI.voxel_edges[:,:,:,0,1]) 
            mask_vox_y2 = (self.POCA_points[i,1]<=self.VOI.voxel_edges[:,:,:,1,1])

            mask_vox_z1 = (self.POCA_points[i,2]>=self.VOI.voxel_edges[:,:,:,0,2]) 
            mask_vox_z2 = (self.POCA_points[i,2]<=self.VOI.voxel_edges[:,:,:,1,2])

            mask_x_ = mask_vox_x1 & mask_vox_x2
            mask_y_ = mask_vox_y1 & mask_vox_y2    
            mask_z_ = mask_vox_z1 & mask_vox_z2

            mask_vox = mask_x_ & mask_y_ & mask_z_
            
            indice = (mask_vox==True).nonzero()
            
            # because of float precision (I assume?), more than 1 vox can be triggered.
            # if it is the case, only keep one voxel
            if(len(indice)==1):
                indices[i] = indice
            elif(len(indice)>1): 
                indices[i] = indice[0]
                
        # Make sure each poca point triggers a voxel
        assert len((indices == -1*torch.ones_like(indices)).nonzero()) == 0
                
        print('DONE')
        return indices.int()
    

    def compute_mask_VOI(self)->Tensor:

        '''
        Only keep events for which POCA point is located INSIDE the VOI.
        
        INPUT: self.poca_points, the POCA locations
        
        OUTPUT: self.POCA_mask, a 1D mask to be applied on events 
        '''
        masks = []
        for i in range(3):
            masks.append((self.POCA_points[:,i]>=self.VOI.xyz_min[i]) & (self.POCA_points[:,i]<=self.VOI.xyz_max[i]))
       
        return (masks[0] & masks[1] & masks[2])
 

    def assign_score_voxel(self, score_feature:Tensor, dtheta_min_max:List[float])->List:

        '''
        Each event for which (dtheta_min<dtheta<dtheta_max) 
        applies a score = score_feature to the voxel the POCA point is located in.

        Must be modified in order to avoid for loop!
        '''
        from fastprogress import progress_bar


        # Create a list with size [Nvox_x,Nvox_y,Nvox_z]
        score_list = torch.zeros((self.VOI.n_vox_xyz[0],self.VOI.n_vox_xyz[1],self.VOI.n_vox_xyz[2])).tolist()

        # Assign an empty list [] to each element of score_list
        print('\nAssigning score to voxels:')
        for i in progress_bar(range(self.VOI.n_vox_xyz[0])):
            for j in range(self.VOI.n_vox_xyz[1]):
                for k in range(self.VOI.n_vox_xyz[2]):
                    score_list[i][j][k]=[]

        # Loop over events for which dtheta>dtheta_min and dtheta<dtheta_max
        for i in progress_bar(range(len(score_feature))):
            mask_dtheta_min_max = (self.tracks.dtheta[i]>dtheta_min_max[0])&(self.tracks.dtheta[i]<dtheta_min_max[1])

            if(mask_dtheta_min_max):
                ix,iy,iz = self.indices[i,0],self.indices[i,1],self.indices[i,2]
                score_list[ix][iy][iz].append(score_feature[i].numpy())
        print('DONE')
        return score_list

    
    def compute_final_voxel_score(self,
                                  score_list:List,
                                  score_method:functools.partial)->Tensor:
        '''
        Computes the final score of every voxel, using score_method as input.

        score_method can be mean, variance, quartile, etc...

        For loop has to be changed!
        '''
        
        from fastprogress import progress_bar

        final_voxel_scores = torch.zeros(self.VOI.n_vox_xyz[0],
                                         self.VOI.n_vox_xyz[1],
                                         self.VOI.n_vox_xyz[2])

        #loop over every voxels
        print('\nComputing final scores:')
        for i in progress_bar(range(self.VOI.n_vox_xyz[0])):
            for j in range(self.VOI.n_vox_xyz[1]):
                for k in range(self.VOI.n_vox_xyz[2]):

                    if(score_list[i][j][k]!=[]):
                            final_voxel_scores[i,j,k] = score_method(score_list[i][j][k])
                    else:
                        final_voxel_scores[i,j,k] = 0.
        print('DONE')
        
        return final_voxel_scores
    
    
    def poca_analysis(self,
                      score_feature:Tuple[torch.tensor,str],
                      dtheta_range:List[float],
                      score_method:functools.partial,
                      save:bool=False,
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
        
        # create output directory
        self.poca_params = {"score_feature":score_feature[1],
                          "score_method":get_partial_name_args(score_method),
                          "dtheta_range":dtheta_range}
        
        self.poca_name = self.get_poca_name()
        self.dir_name = self.output_dir + self.poca_name + "/"
        self.create_directory(dir_name=self.dir_name)
        
        # Analysis
        score_list = self.assign_score_voxel(score_feature=score_feature[0],
                                             dtheta_min_max=dtheta_range)
        
        final_scores = self.compute_final_voxel_score(score_list=score_list,
                                                      score_method=score_method)
        
        if save:
            with open(self.dir_name+"poca_final_scores", "wb") as f:
                pickle.dump(final_scores,f)
                
        if plot:
            from plotting.Plotting import plot_VOI_pred
            plot_VOI_pred(preds=final_scores, filename=self.dir_name+"poca_plot")
            
        return final_scores
       
    
    def get_poca_name(self,) -> str:
        
        feature = "feature_{}_".format(self.poca_params["score_feature"])
        method = "method_{}_".format(self.poca_params["score_method"])
        dtheta = "{:.2f}_{:.2f}_rad".format(self.poca_params["dtheta_range"][0],self.poca_params["dtheta_range"][1])
                
        return feature+method+dtheta
    
    
    def save(self,directory:str) -> None:

        import pickle

        with open(directory+"poca_class", "wb") as f:
            pickle.dump(self,f)

            
    def generate_csv_file(self, filename:str)->None:

        df = pd.DataFrame()

        df['theta_in'] = self.tracks.theta_in
        df['theta_out'] = self.tracks.theta_out

        df['theta_x_in'] = self.tracks.theta_x_in
        df['theta_x_out'] = self.tracks.theta_x_out
        df['theta_y_in'] = self.tracks.theta_y_in
        df['theta_y_out'] = self.tracks.theta_y_out

        df['E'] = self.tracks.E

        df['x_poca'] = self.POCA_points[:,0]
        df['y_poca'] = self.POCA_points[:,1]
        df['z_poca'] = self.POCA_points[:,2]
        df['x_in'] = self.tracks.xyz_in[0]
        df['y_in'] = self.tracks.xyz_in[1]
        df['z_in'] = self.tracks.xyz_in[2]
        df['x_out'] = self.tracks.xyz_out[0]
        df['y_out'] = self.tracks.xyz_out[1]
        df['z_out'] = self.tracks.xyz_out[2]
        
        df.to_csv(filename + '.csv',index=False)


    def compute_voxel_std(self,) -> torch.tensor:
        
        voxel_std = torch.zeros([self.VOI.n_vox_xyz[0],
                                 self.VOI.n_vox_xyz[1],
                                 self.VOI.n_vox_xyz[2]])
        
        for i in range(self.VOI.n_vox_xyz[0]):
            for j in range(self.VOI.n_vox_xyz[1]):
                for k in range(self.VOI.n_vox_xyz[2]):
                    
                    voxel_std = torch.std(dtheta)
                    
                    
    def reweight(self,voxel_std:torch.tensor) -> torch.tensor:
        
        voxel_std = voxel_std - torch.min(voxel_std) / (torch.max(voxel_std) - torch.min(voxel_std))