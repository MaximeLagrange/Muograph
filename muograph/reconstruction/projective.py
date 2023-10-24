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


class Projective():

    def __init__(self, tracks:Tracking=None, 
                 voi:VolumeInterest=None, 
                 from_save:str=None, 
                 angular_res:float=0., 
                 spatial_res:float=0.):

        r"""
        Class used for projective reconstruction method for absorption muography. The algorithm is based on 
        "A projective reconstruction method for underground or hidden structures using atmospheric muons 
        absorption data" - L. Bonechi et al 2015 JINST 10 P02003

        Arguments:
        - tracks: Tracking
        - voi: VolumeInterest
        - from_save: str, if not None read triggered voxels saved in the file.
        """

        if((tracks is not None) & (voi is not None)):
            self.tracks = tracks
            self.voi = voi

        #Detector features
        self.angular_res = angular_res
        self.spatial_res = spatial_res
        
    def get_2d_hist_from_tracks(self,d:float):

        return
    
    def get_binning(self,d:float):

        #bin size
        bs = self.spatial_res + d*self.angular_res

        return
    


