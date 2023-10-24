# Usual suspects
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from typing import Dict, List, Union, Tuple, Optional
import skspatial
from fastprogress import progress_bar
from functools import partial
import os
from os import getcwd


# Muograph
from volume.Volume_Interest import VolumeInterest
from detector.pixel_detector import PixelDetector
from utils.read_file import hits_csv_to_tensor

class AbsorptionAnalysis():

    r"""
    Class used for muon absorption tomography.

    """

    def __init__(self,
                 filename:str,
                 directory:str,
                 CRY_date:str,
                 out_filename_title:str,
                 n_gen_event:int,
                 partial_detector:partial=None,
                 energy_cut:List[int]=None,
                 ev_per_batch:int=50000,
                 out_filename_directory:str=None,) -> None:
        
        # Cut on energy
        self.energy_cut = energy_cut
        
        # Input data filename
        self.input_filename = filename
        self.input_directory = directory

        # Output data filename
        self.n_gen_event = n_gen_event # in millions
        self.CRY_date = CRY_date
        out_filename_title += "_{}M_gen_event".format(self.n_gen_event)
        if(energy_cut is not None):
            out_filename_title += "_{}to{}_MeV".format(self.energy_cut[0],self.energy_cut[1])

        self.out_directory = os.path.abspath("absorption_analysis.py").replace("dev/absorption_analysis.py","analysis/absorption/"+out_filename_title+"/")

        if(out_filename_directory is not None):
            self.out_directory = out_filename_directory

        # Create output directory
        self.make_output_directory(self.out_directory)

        # #event per batch
        self.ev_per_batch = ev_per_batch

        # Detector
        self.partial_detector = partial_detector

        # Get hits
        #hit positions in mm, energy in MeV/c2
        self.hits,self.E = hits_csv_to_tensor(filename=self.input_filename,directory=self.input_directory)
        print("\nInput data file = {} \nInput directory = {}".format(self.input_filename,
                                                                     self.input_directory))
        print("\n# event = {}".format(self.hits.size()[-1]))

        # Apply cut on energy
        if(self.energy_cut is not None):
            self.hits,self.E = self.apply_energy_cut(self.energy_cut)
            print("\n# event = {} within energy range [{},{}] MeV/c2".format(self.hits.size()[-1],
                                                                             self.energy_cut[0],
                                                                             self.energy_cut[1]))
            
        # Save true hits
        with open(self.out_directory+"true_hits","wb") as f:
            pickle.dump(self.hits,f)

        # Save true energy
        with open(self.out_directory+"true_energy","wb") as f:
            pickle.dump(self.E,f)

        self.hit_files, self.E_files = self.divide_data_file()

        # Get and save reconstructed hits
        self.rec_hits_files = None
        if(self.partial_detector is not None):
            self.rec_hits_files, self.rec_E_files = self.get_and_save_rec_hits()

        # Tracking
        if(self.partial_detector is not None):
            pass




    def get_and_save_tracking_from_rec_hits(self) -> List[str]:

        for filename, E_filename in zip(self.rec_hits_files,self.rec_E_files):
            with open(self.out_directory+filename, "rb") as f:
                hits = pickle.load(f)

            with open(self.out_directory+E_filename, "rb") as f:
                E = pickle.load(f)

        return 
    
    def get_and_save_rec_hits(self) -> Tuple[List[str]]:
        
        rec_hits_files, rec_E = [],[]
        for hit_file,i in zip(self.hit_files,range(len(self.hit_files))):
            # Compute reconstructed hits
            det = self.partial_detector(hit_file)

            # Save hits after full coincidence trigger
            filename = self.get_rec_hits_filename() + "_{}of{}".format(i+1,len(self.hit_files))
            with open(self.out_directory+filename,'wb') as f:
                pickle.dump(det.rec_hits, f)
            rec_hits_files.append(filename)
            print("rec_hits filename = ",filename)

            # Save energies after full coincidence trigger
            E_filename = "E_{}of{}".format(i+1,len(self.hit_files))
            with open(self.out_directory+E_filename,'wb') as f:
                pickle.dump(self.E_files[i][det.full_co_mask], f)
            rec_E.append(E_filename)
            
        return (rec_hits_files,rec_E)

    def make_output_directory(self, out_directory:str) -> None:

        if(os.path.exists(out_directory)):
            awnser=None
            while(awnser!=0):
                awnser=input("\nOverwrite {} ? [y,n]".format(out_directory))
                if(awnser=="n"):
                    raise Exception("\nAnalysis cancelled, choose another out_filename")
                if(awnser=="y"):
                    break
                else:
                    print('y = yes, n = no')
        else:
            os.makedirs(out_directory)
            print("Create directory {}".format(out_directory))

    def get_rec_hits_filename(self) -> str:
        
        ev_label = "{}kevents".format(int(self.ev_per_batch/1000))
        
        return self.CRY_date+"_"+ev_label+"_"+"rec_hits"    

    def divide_data_file(self) -> Tuple[List[Tensor]]:
    
        r"""
        INPUTS:
        - hits:Tensor with size (3,n_plane,n_event) (self.hits) in mm
        - E:Tensor with size (n_event), muon's energy (self.E) in MeV/c2
        - ev_per_batch:int The number of event used per tensor. Can be adjusted 
        depending on your computed power. (self.ev_per_batch)
        
        OUTPUT:
        - hit_files,E_files: Tuple[List[Tensor]]
        """
        
        n_event = self.hits.size()[-1]
        n_file = n_event//self.ev_per_batch
        if((n_event%self.ev_per_batch)!=0):
            n_file += 1
            
        hit_files,E_files = [],[]
    
        for i in range(n_file):
            hit_files.append(self.hits[:,:,(i*self.ev_per_batch):((i+1)*self.ev_per_batch)])
            E_files.append(self.E[(i*self.ev_per_batch):((i+1)*self.ev_per_batch)])

        return hit_files,E_files


    def apply_energy_cut(self, energy_cut:List[int]) -> Tuple[Tensor]:
        r"""
        Apply cut on data file based on the generated muon energy.
        
        INPUTS:
        - hits:Tensor with size (3,n_plane,n_event) (self.hits)
        - E:Tensor with size (n_event), muon's energy in MeV (self.E)
        - energy_cuts:List[int] The lower and upper cut on energy [Emin,Emax] in MeV
        """
            
        mask = (self.E>=energy_cut[0])&(self.E<=energy_cut[1])
        return self.hits[:,:,mask],self.E[mask] 


    # def get_rec_hits(self) -> None:
        
    #     try:
    #         f = open(self.out_directory+filename)
    #     except:
    #         #create a new file!
    #     else:
    #         #a file already exist
    #         # do you want to overwrite?

    #     for hit_file,E_file,i in progress_bar(zip(self.hit_files,self.E_files,range(len(self.hit_files)))):

    #         self.partial_detector(hit_file)

    #         filename = self.get_rec_hits_filename()
    #         filename += "{}of{}".format(i,len(self.hit_files))

            
    #         with open(self.out_directory+filename) as f:


