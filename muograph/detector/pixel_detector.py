import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from typing import Dict, List, Union, Tuple, Optional
import skspatial
from fastprogress import progress_bar


class PixelDetector():

    def __init__(self,
                 hits:Tensor,
                 xy_min_max:Tuple[float], 
                 pixel_width:float=9.9, 
                 gap:float=0.1, 
                 pixel_efficiency:float=1.) -> None:

        r"""
        Class for pixel detector modeling. Each panel is assumed to have the same xy position. 
        Each panel has the same characteristics. Pixels are symmetrical (x & y).
        
        INPUT:
         - xy_max_min: (x_min, x_max, y_min, y_max) xy position of the detector plane angle 
         - pixel_width: The width (=height) of each pixel in [mm]
         - gap: The gap between each pixels (in x & y)
         - pixel_efficency: The efficiency of each pixel (not implemented yet)

        """
        # Detector
        self.n_plane  = hits.size()[1]

        # Detector panel size
        self.xy_min_max = xy_min_max
        self.x_range = abs(self.xy_min_max[0] - self.xy_min_max[1])
        self.y_range = abs(self.xy_min_max[2] - self.xy_min_max[3])

        # Pixels properties
        self.pixel_width = pixel_width
        self.gap = gap
        self.pixel_efficiency = pixel_efficiency
        self.n_pixels_x = int(self.x_range/(self.pixel_width+self.gap))
        self.n_pixels_y = int(self.y_range/(self.pixel_width+self.gap))

        # Pixels centers and edges
        self.pixel_centers, self.pixel_edges = self.compute_pixel_centers_edges()

        # Muon hits
        self.hits = hits

        # Reconstructed hits
        self.rec_hits, self.n_hit = self.find_triggered_pixels()
        self.full_co_mask = self.compute_full_co_mask()
        self.rec_hits = self.rec_hits[:,:,self.full_co_mask]


    def compute_pixel_centers_edges(self) -> Tuple[Tensor]:

        r"""
        Compute pixels centers and edges positions.

        INPUT:
         - n_pixel_x,y: The number of pixels along the x,y dimension.
         - gap: The gap between each pixel.
         - pixel_width
         - xy_min_max

        OUTPUT:
         - pixel_centers: Tensor with size [n_pixel_x,n_pixel_y,2] 
         containing the xy position of each pixel.

         - pixel_edges: Tensor with size [n_pixel_x,n_pixel_y,2,2] 
         edecontaining the xy position of the bottom left corner (pixel_edges[ix,iy,0,:])
        and upper right corner (pixel_edges[ix,iy,1,:]) of each pixel.
        """

        # Pixels centers
        pixel_centers = torch.zeros((self.n_pixels_x,self.n_pixels_y,2))

        print("\nPixel generation")
        for i in progress_bar(range(self.n_pixels_x)):
            for j in range(self.n_pixels_y):
                pixel_centers[i,j,0] = self.pixel_width/2 + i*self.gap + i*self.pixel_width + self.xy_min_max[0]
                pixel_centers[i,j,1] = self.pixel_width/2 + j*self.gap + j*self.pixel_width + self.xy_min_max[2]
                
        # Pixels edges
        pixel_edges = torch.zeros((self.n_pixels_x,self.n_pixels_y,2,2))
        pixel_edges[:,:,0,:] = pixel_centers[:,:,:]-self.pixel_width/2
        pixel_edges[:,:,1,:] = pixel_centers[:,:,:]+self.pixel_width/2

        print("DONE")
        return pixel_centers, pixel_edges

    
    def find_triggered_pixels(self):

        r"""
        Find the indices of pixels triggered by a muon hit. The reconstructed hit position is computed as:

            x_rec = x_pixel_center + x_smearing
            y_rec = x_pixel_center + y_smearing

        x_smearing is sampled from a uniform distribution ranging from -pixel_width/2 to pixel_width/2

        Pixel efficiency is not implemented yet.

        INPUT:
         - pixel_edges: Tensor with size [n_pixel_x,n_pixel_y,2,2]
         - hits: Tensor with size [3,n_plane,n_event]

        OUTPUT:
         - n_hit: Tensor with size [n_event]. Contains the number of detector planes hit for a given event.
         - rec_hits: Tensor with size [3,n_plane,n_event]. 
         For now, we only considerate event for which n_hit = n_plane (full coincidence events).
        """

        n_hit = torch.zeros((len(self.hits[0,0])))
        rec_hits = torch.zeros_like(self.hits)

        print("\n Pixel triggering")
        pixel_edges_d = self.pixel_edges[:,:,:,None,:].expand(-1,-1,-1,self.n_plane,-1)

        for event in progress_bar(range(self.hits.size()[-1])):

            mask_x = (pixel_edges_d[:,0,0,:,0]<self.hits[0,:,event]) & (pixel_edges_d[:,0,1,:,0]>self.hits[0,:,event]) 
            mask_y = (pixel_edges_d[0,:,0,:,1]<self.hits[1,:,event]) & (pixel_edges_d[0,:,1,:,1]>self.hits[1,:,event])  

            n_hit[event] = np.min((len(mask_x.nonzero()),len(mask_y.nonzero())))

            indices = torch.zeros((self.n_plane,3))
            if(n_hit[event]==self.n_plane):
                indices[:,0]=mask_x.nonzero()[:,0]
                indices[:,1]=mask_y.nonzero()[:,0]
                indices[:,2]=mask_y.nonzero()[:,1]

                for plane in range(self.n_plane):
                    for coord in range(2):
                        rec_hits[coord,plane,event] = self.pixel_centers[int(indices[indices[:,2]==plane][0][coord].item()),0,0] + np.random.uniform(low = -self.pixel_width/2,
                                                                                                                                            high = self.pixel_width/2,
                                                                                                                                            size = 1)
                    rec_hits[2,plane,event] = self.hits[2,plane,event]

        return rec_hits, n_hit

    
    def compute_full_co_mask(self):

        r"""
        Compute the full coincidence mask (a muon leaves a hit in EVERY detector panel)

        OUTPUT:
        - Bollean tensor with size [n_event]
        """

        return (self.n_hit==self.n_plane)
    
    
    def plot_detector_panel(self, pixel_indices:List[Tensor]=None,filename:str=None):
            
        scale = 1.6
        fig,ax = plt.subplots(figsize=(scale*10,scale*7))
        fig.suptitle("Pixel detector panel {}x{} pixels\nPixel size = {} [mm]\nGap = {} [mm]".format(self.n_pixels_x,
                                                                                                    self.n_pixels_y,
                                                                                                    self.pixel_width,
                                                                                                    self.gap))
        for i in range(self.n_pixels_x):
            for j in range(self.n_pixels_y):
                x = self.pixel_centers[i,j,0]
                y = self.pixel_centers[i,j,1]
                rectangle = plt.Rectangle((x-self.pixel_width/2,y-self.pixel_width/2), 
                                        self.pixel_width, 
                                        self.pixel_width,
                                        ec="red",
                                        alpha=.3)       
                plt.gca().add_patch(rectangle)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')

        if(pixel_indices is not None):
            
            for indices in pixel_indices:
                ix,iy = indices[0],indices[1]
                x = self.pixel_centers[ix,iy,0]
                y = self.pixel_centers[ix,iy,1]
                rectangle = plt.Rectangle((x-self.pixel_width/2,y-self.pixel_width/2), 
                                        self.pixel_width, 
                                        self.pixel_width,
                                        fc="green",
                                        ec='green',
                                        alpha=.3)
                plt.gca().add_patch(rectangle)

        plt.axis('scaled')
        if(filename is not None):
            plt.savefig(filename)
        plt.show()
