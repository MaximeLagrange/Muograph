import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from typing import Dict, List, Union, Tuple, Optional

from IPython.display import display, Math

    
class VolumeInterest():

    def __init__(self,position:Tuple[float],dimension:Tuple[float],voxel_width:float=10):

        '''
        position = [x,y,z] in mm
        dimension = [dx,dy,dz] in mm 
        Voxel width = 10 mm default
        '''

        # VOI position
        self.xyz = torch.tensor(position)
        
        # VOI dimensions
        self.dxyz = torch.tensor(dimension)
        self.dxyz = torch.tensor(dimension)

        self.xyz_min = self.xyz - self.dxyz/2
        self.xyz_max = self.xyz + self.dxyz/2

        # Voxel width
        self.vox_width = voxel_width

        # Voxelization
        self.n_vox_xyz = self.Compute_N_voxel()
        self.voxel_centers,self.voxel_edges = self.Generate_voxels()

        # True X0
        self.x0_true = None
        self.density_map_true = None

        # Pred X0
        self.x0_pred = None
        self.density_map_pred = None

    # def __repr__(self) -> str:
    #     display(Math(r'VOI \:\: at \:\:x,y,z = {:.2f},{:.2f},{:.2f}\:\:in\:\:[mm]'.format(self.xyz[0],
    #     self.xyz[1],
    #     self.xyz[2])))

    #     display(Math(r'VOI \:\: has \:\:dimension \:\:dx,dy,dz = {:.2f},{:.2f},{:.2f}\:\:in\:\:[mm]'.format(self.dxyz[0],
    #     self.dxyz[1],
    #     self.dxyz[2])))

    #     display(Math(r'VOI \:\: \# \:\:voxel \:\:along \:\:x,y,z = {:.2f},{:.2f},{:.2f}'.format(self.n_vox_xyz[0],
    #     self.n_vox_xyz[1],
    #     self.n_vox_xyz[2])))

    #     display(Math(r'VOI \:\: voxel \:\:size = {:.2f}\:\:[mm]'.format(self.vox_width)))


    def Compute_N_voxel(self):

        nx = self.dxyz[0]/self.vox_width
        ny = self.dxyz[1]/self.vox_width
        nz = self.dxyz[2]/self.vox_width

        if((nx%1!=0)|(ny%1!=0)|(nz%1!=0)):
            print('ERROR')
            print('Voxel size does not match VOI dimensions')
            print('Please make sure that dimension / voxel_width = integer')
        return np.array([int(nx),int(ny),int(nz)])


    def Compute_voxel_centers(self,x_min_: float, 
                              x_max_: float,
                              Nvoxel_: int) -> torch.tensor:
                                    
        '''
        x_min,max border of the volume of interset for a given coordinate
                
        return voxels centers position along given coordinate
        '''
        xs_ = torch.linspace(x_min_,x_max_,Nvoxel_+1)
        xs_ += self.vox_width/2
        return xs_[:-1]    


    def Generate_voxels(self)->Tensor:

        # self.voxel_centers.size() = [Nvox_x,Nvox_y,Nvox_z,3]
        # self.voxel_edges.size() =  [Nvox_x,Nvox_y,Nvox_z,2,3]
            
        voxels_centers = torch.zeros((self.n_vox_xyz[0],self.n_vox_xyz[1],self.n_vox_xyz[2],3),dtype=torch.double)
        
        xs_ = self.Compute_voxel_centers(x_min_=self.xyz_min[0], x_max_=self.xyz_max[0],
                                    Nvoxel_= self.n_vox_xyz[0])
        ys_ = self.Compute_voxel_centers(x_min_=self.xyz_min[1], x_max_=self.xyz_max[1],
                                    Nvoxel_= self.n_vox_xyz[1])
        zs_ = self.Compute_voxel_centers(x_min_=self.xyz_min[2], x_max_=self.xyz_max[2],
                                    Nvoxel_= self.n_vox_xyz[2])
                    
        for i in range(len(ys_)):
            for j in range(len(zs_)):
                voxels_centers[:,i,j,0]=xs_
        for i in range(len(xs_)):
            for j in range(len(zs_)):
                voxels_centers[i,:,j,1]=ys_
        for i in range(len(xs_)):
            for j in range(len(ys_)):
                voxels_centers[i,j,:,2]=zs_
        
        
        voxels_edges = torch.zeros((self.n_vox_xyz[0],self.n_vox_xyz[1],self.n_vox_xyz[2],2,3))

        voxels_edges[:,:,:,0,:] = voxels_centers-self.vox_width/2
        voxels_edges[:,:,:,1,:] = voxels_centers+self.vox_width/2

        return voxels_centers, voxels_edges


    def load_rad_length(self,rad_length:Optional[float]=None):

        self.X0 = torch.zeros(([self.n_vox_xyz[0],self.n_vox_xyz[1],self.n_vox_xyz[2]]))+rad_length



    


