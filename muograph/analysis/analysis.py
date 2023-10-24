# usual suspects
import pickle
import functools
from typing import List, Union
NoneType = type(None)

# Muograph
from reconstruction.binned_clustered import BCA
from reconstruction.POCA import POCA
from reconstruction.ASR import ASR


from volume.Volume_Interest import VolumeInterest

class analysis:
    
    def __init__(self,
                 VOI:VolumeInterest,
                 compute_tracks:bool=False,
                 n_event:Union[int,NoneType]=None,
                 input_dir:str="../data/nuclear_waste_container/G4_data/",
                 output_dir:str="../data/nuclear_waste_container/",
                 analysis_type:str="scattering"):
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        import os
        if(os.path.exists(self.output_dir)==False):
            os.mkdir(self.output_dir)
            print("\n{} directory created".format(self.output_dir))
            
        self.VOI = VOI
        
        # Get input files name
        import os
        list_dir = os.listdir(self.input_dir)
        self.csv_filenames = [self.input_dir+file for file in list_dir if  os.path.splitext(file)[-1]==".csv"]
        assert self.csv_filenames != [], input_dir+" contains no csv file!"
        
        # read files
        from utils.read_file import hits_csv_to_tensor
        hits,E = hits_csv_to_tensor(filename=self.csv_filenames,n_event=n_event)
        
        # tracking
        if(compute_tracks):
            from tracking.Tracking import Tracking
            tracks = Tracking(hits=hits,voi=self.VOI,E=E,analysis=analysis_type)
            tracks.save(self.output_dir+"tracks")
            del tracks
            
        
    def bca_setup(self,) -> None:
        # read trackfile
        with open(self.output_dir+"tracks","rb") as f:
            tracks = pickle.load(f)
        
        bca = BCA(tracks=tracks,output_dir=self.output_dir)
        bca.save(self.output_dir)
        del bca
        
    def poca_setup(self,) -> None:
        with open(self.output_dir+"tracks","rb") as f:
            tracks = pickle.load(f)
            
        poca = POCA(output_dir=self.output_dir,tracks=tracks)
        poca.save(self.output_dir)
        del poca
    
    def asr_setup(self,) -> None:
        with open(self.output_dir+"tracks","rb") as f:
            tracks = pickle.load(f)
            
        asr = ASR(tracks=tracks,output_dir=self.output_dir)
        asr.save(self.output_dir,triggered_vox_only=True)
        del asr
        
    def load_pickle(self,keyword:str,file_path:str=None):
        
        if(file_path is None):
            with open(self.output_dir+keyword,"rb") as f:
                return pickle.load(f)
        else:
            with open(file_path,"rb") as f:
                return pickle.load(f)
            