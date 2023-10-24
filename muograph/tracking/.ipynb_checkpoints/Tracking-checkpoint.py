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

# Muograph imports
from volume.Volume_Interest import *
from utils.Special_relativity import momentum_from_kinetic_E

class Tracking():

    r"""
    Tracking class computes muon features relevant to image reconstruction:

    -  dtheta: Tensor[float] with size [n_event], the scattering angle
    -  dtheta_x, dtheta_y:  Tensor[float] with size [n_event], 
    the projected scattering angle.
    - theta_in, theta_out, phi_in, phi_out: Tensor[float] with size [n_event],
    INcoming, OUTgoing zenith/azimuthal angle.
    - E: Tensor[float] with size [n_event], muon's energy
    - xyz_in_VOI, xyz_out_VOI: Tensor[float] with size [],
    - xyz_discrete_in: Tensor[float] with size, 
    """

    def __init__(self, hits:Tensor, voi:VolumeInterest=None, E=None, analysis:str='scattering'):

        '''
        hits:Tensor with size [Ncoord=3,Nplane,Nevent]

        Arguments:
        - abs: If abs = True, only one tracks per event will be reconstructed using every hits 
               (Absorption Muography)

               If abs = False, two tracks per event will be reconstructed using upper and lower detector hits.
               (Scattering Muograpy) 
        '''

        self.hits = hits
        self.n_event = int(self.hits.shape[2])
        self.n_event_masked = None
        self.events = torch.linspace(0,self.n_event-1,self.n_event)
        self.analysis = analysis
        if((analysis!='scattering') & (analysis!='absorption')):
            raise RuntimeError("Choose appropriate analysis mode:\n  - scattering\n  - absorption"
)
        # Detector
        self.n_plane = int(self.hits.shape[1])
        if voi is not None:
            self.voi = voi

        #Tracking Scattering
        if(self.analysis=='scattering'):
            # Tracks
            # self.tracks_up,self.tracks_down,self.point_up,self.point_down = self.Compute_tracks_new()
            self.tracks_up,self.tracks_down,self.point_up,self.point_down = self.Compute_tracks_scat()

            # Scattering angle
            self.dtheta, self.dtheta_x, self.dtheta_y = self.Compute_scattering_angle()

            # Angles
            self.theta_in, self.theta_out, self.theta_x_in, self.theta_x_out, self.theta_y_in, self.theta_y_out = self.Compute_theta_in_out()
            self.phi_in, self.phi_out = self.Compute_phi_in_out()

            # xyz_in
            if voi is not None:
                # self.xyz_in, self.xyz_out = self.Compute_xyz_in_out()
                self.xyz_in_VOI, self.xyz_out_VOI = self.Compute_xyz_in_out_new()
                self.xyz_discrete_in, self.xyz_discrete_out = self.Compute_discrete_tracks_scat()
        
        # Tracking absorption
        elif(self.analysis=='absorption'):
            # Tracks
            self.tracks, self.point = self.Compute_tracks_abs()
            self.theta, self.theta_x, self.theta_y = self.Compute_theta()
            self.phi = self.Compute_phi()

            # xyz_VOI
            self.xyz_in_VOI = self.Compute_xyz_in_new()
            self.xyz_discrete_in = self.Compute_discrete_tracks_abs()
        
        # Energy
        if E is not None:
            self.E = torch.tensor(np.array(E)) #in MeV/c2
            self.p = momentum_from_kinetic_E(self.E,mass=105.7) # in MeV/c
        else:
            self.E = None
            self.p = None

        # mask
        self.mask = torch.where(torch.ones(self.n_event)==1,True,False)

    def Compute_tracks_scat(self)->Tensor:

        '''
            return upper and lower point and tracks computed from hits[3][Nplane][Nevent]

            upper tracks, lower tracks have size [Nevent,3]
            upper point, lower point have size [Nevent,3]

        '''
        from skspatial.objects import Line
        from skspatial.objects import Points
        from skspatial.plotting import plot_3d
        from fastprogress import progress_bar


        track_up,track_down = torch.zeros(self.n_event,3,dtype=torch.double),torch.zeros(self.n_event,3,dtype=torch.double)
        point_up,point_down = torch.zeros(self.n_event,3,dtype=torch.double),torch.zeros(self.n_event,3,dtype=torch.double)

        if(self.n_plane%2!=0):
            print('Current tracking implementation only allow for a detector having the same number of upper and lower detection planes')
        
        print('\nTracking in progress')
        for i in progress_bar(range(self.n_event)):
            up_hits = torch.transpose(self.hits[:,:int(self.n_plane/2),i],0,1) #4
            down_hits = torch.transpose(self.hits[:,int(self.n_plane/2):,i],0,1) #4
            
            pointsUP, pointsDOWN = Points(up_hits),Points(down_hits)
            fit_UP, fit_DOWN = Line.best_fit(pointsUP),Line.best_fit(pointsDOWN)
            track_up[i], track_down[i] = torch.tensor(fit_UP.direction),torch.tensor(fit_DOWN.direction)
            point_up[i], point_down[i] = torch.tensor(fit_UP.point),torch.tensor(fit_DOWN.point)

        print('Tracking done')
        
        return track_up,track_down,point_up,point_down
    

    def Compute_tracks_abs(self)->Tensor:

        '''
            return points and tracks computed from hits[3][Nplane][Nevent]

            tracks have size [Nevent,3]
            points have size [Nevent,3]

        '''
        from skspatial.objects import Line
        from skspatial.objects import Points
        from skspatial.plotting import plot_3d
        from fastprogress import progress_bar

        tracks, points = torch.zeros(self.n_event,3,dtype=torch.double), torch.zeros(self.n_event,3,dtype=torch.double)

        print('\nTracking in progress')
        for i in progress_bar(range(self.n_event)):
            hits = torch.transpose(self.hits[:,:,i],0,1) #4
            
            ps = Points(hits)
            fit= Line.best_fit(ps)
            tracks[i] = torch.tensor(fit.direction)
            points[i] = torch.tensor(fit.point)

        print('Tracking done')
        
        return tracks,points
    
    def Compute_tracks_new(self)->Tuple[Tensor]:

        '''
            return upper and lower point and tracks computed from hits[3][Nplane][Nevent]

            upper tracks, lower tracks have size [Nevent,3]
            upper point, lower point have size [Nevent,3]

        '''

        from fastprogress import progress_bar


        track_coeff = torch.zeros((2,4,self.n_event))
        res = torch.zeros((2,2,self.n_event))
        points_, tracks_ = torch.zeros(2,3,self.n_event),torch.zeros(2,3,self.n_event)


        n_plane_min = [0,int(self.n_plane/2)]
        n_plane_max = [int(self.n_plane/2),self.n_plane]

        for plane_min,plane_max,j in zip(n_plane_min, n_plane_max,[0,1]):

            for i in progress_bar(range(self.n_event)):
                ls_x = np.linalg.lstsq(np.vstack([self.hits[0,plane_min:plane_max,i], 
                                                np.ones(len(self.hits[0,plane_min:plane_max,i]))]).T, 
                                    self.hits[2,plane_min:plane_max,i],rcond=None)[:2]
                
                ls_y = np.linalg.lstsq(np.vstack([self.hits[1,plane_min:plane_max,i], 
                                                np.ones(len(self.hits[1,plane_min:plane_max,i]))]).T, 
                                    self.hits[2,plane_min:plane_max,i],rcond=None)[:2]

                # Track Up
                track_coeff[j,0,i]=ls_x[0][0]
                track_coeff[j,1,i]=ls_x[0][1]

                track_coeff[j,2,i] = ls_y[0][0]
                track_coeff[j,3,i] = ls_y[0][1]

                res[j,0,i] = ls_x[1][0]
                res[j,1,i] = ls_y[1][0]
            
            points_[j,0], points_[j,1], points_[j,2] = -track_coeff[j,1]/track_coeff[j,0],-track_coeff[j,3]/track_coeff[j,2],0
            tracks_[j,0], tracks_[j,1], tracks_[j,2] = 1/track_coeff[j,0], 1/track_coeff[j,2],1

        return torch.transpose(tracks_[0],0,1),torch.transpose(tracks_[1],0,1),torch.transpose(points_[0],0,1),torch.transpose(points_[1],0,1)

    def Compute_scattering_angle(self)->Tensor:


        def norm(vect: Tensor)->Tensor:
            '''
            Compute vector norm. Vector must have shape [Nevent][3] (x,y,z) or [3].
            '''
            if((vect.size(dim=-1)==3) & (len(vect)==3)):
                x,y,z = vect[0],vect[1],vect[2]
            else:
                x,y,z = vect[:,0],vect[:,1],vect[:,2]
            return torch.sqrt(x**2+y**2+z**2)

        '''
        Compute spatial scattering angle dtheta, and x,y projected scattering angle dtheta_x, dtheta_y, for each event.
        INPUT: tracks_up, tracks_down
        OUTPUT: dtheta_x,dtheta_y,dtheta
        '''
        
        xup, yup, zup = self.tracks_up[:,0],self.tracks_up[:,1],self.tracks_up[:,2]
        xdown, ydown, zdown = self.tracks_down[:,0],self.tracks_down[:,1],self.tracks_down[:,2]
        
        dtheta = torch.acos((xup*xdown + yup*ydown + zup*zdown)/(norm(self.tracks_up)*norm(self.tracks_down)))
        dtheta_x = torch.acos((xup*xdown + zup*zdown)/((torch.sqrt(xup**2+zup**2))*(torch.sqrt(xdown**2+zdown**2))))
        dtheta_y = torch.acos((yup*ydown + zup*zdown)/((torch.sqrt(yup**2+zup**2))*(torch.sqrt(ydown**2+zdown**2))))
        # scattering amgles close to zero computed as NaN
        #Nans are changed to 0 scattering angles
        dtheta[torch.isnan(dtheta)]=0.0001
        #Random error during track fitting occurs and change z track coordinate to -z
        #which gives an angle close to pi instead of close to zero
        #We change these angles to pi - theta
        dtheta[dtheta==0] = 0.0001
        dtheta[dtheta>2] = torch.abs(torch.pi-dtheta[dtheta>2])
        dtheta_x[dtheta_x>2] = torch.abs(torch.pi-dtheta_x[dtheta_x>2])
        dtheta_y[dtheta_y>2] = torch.abs(torch.pi-dtheta_y[dtheta_y>2])
        dtheta_x[torch.isnan(dtheta_x)]=0.0001
        dtheta_y[torch.isnan(dtheta_y)]=0.0001

        print('Scattering angles computation done')
        return dtheta,dtheta_x,dtheta_y


    def Compute_theta_in_out(self)->Tensor:

        '''
        Compute zenith angle and projected zenith angles along x and y for incoming and outgoing tracks.
        INPUT: tracks_up, tracks_down
        OUTPUT: thetas_in, thetas_out, theta_x_in theta_x_out, theta_y_in, theta_y_out
        '''
        thetas, thetas_x, thetas_y = [], [], []

        for tracks in [self.tracks_up,self.tracks_down]:

            x, y, z = tracks[:,0],tracks[:,1],tracks[:,2]
            r = torch.sqrt(x**2 +y**2+z**2)
            thetas_x.append(torch.atan(x/z))
            thetas_y.append(torch.atan(y/z))
            thetas_ = torch.acos(z/r)
            thetas_ = torch.where(math.pi-thetas_<thetas_, math.pi-thetas_,thetas_)
            thetas.append(thetas_)

        return thetas[0],thetas[1],thetas_x[0], thetas_x[1], thetas_y[0], thetas_y[1]


    def Compute_theta(self)->Tensor:

        '''
        Compute zenith and and projected zenith angle along x and y as follow:

            thetas_x = atan(x/z)
            thetas_y = atan(y/z)
            thetas = acos(z/r)

        INPUT: tracks        
        OUTPUT: thetas, theta_x, theta_y
        '''

        x, y, z = self.tracks[:,0],self.tracks[:,1],self.tracks[:,2]
        r = torch.sqrt(x**2 +y**2+z**2)

        return torch.acos(z/r), torch.atan(x/z), torch.atan(y/z)

    def Compute_phi_in_out(self)->Tensor:

        '''
        Compute azimuthal angle (phi) for incoming and outgoing tracks.
        INPUT: tracks_up, tracks_down
        OUTPUT: phi_in, phi_out
        '''
        phis = []
        for track in [self.tracks_up,self.tracks_down]:

            x, y, z = track[:,0],track[:,1],track[:,2]

            phi = torch.tensor(np.shape(track[:,0]),dtype=torch.double)

            mask1 = (x>0)
            mask2 = (x<0)&(y>=0)
            mask3 = (x<0)&(y<0)
            mask4 = (x==0)&(y>0)
            mask5 = (x==0)&(y<0)

            phi = torch.where(mask1,torch.atan(y/x),phi)
            phi = torch.where(mask2,torch.atan(y/x) + math.pi,phi)
            phi = torch.where(mask3,torch.atan(y/x) - math.pi,phi)
            phi = torch.where(mask4,math.pi/2,phi)
            phi = torch.where(mask5,-math.pi/2,phi)

            phis.append(phi)

        return phis[0],phis[1]
    
    def Compute_phi(self)->Tensor:

        '''
        Compute azimuthal angle (phi).
        INPUT: tracks
        OUTPUT: phi
        '''

        x, y, z = self.tracks[:,0],self.tracks[:,1],self.tracks[:,2]

        phi = torch.tensor(np.shape(self.tracks[:,0]),dtype=torch.double)

        mask1 = (x>0)
        mask2 = (x<0)&(y>=0)
        mask3 = (x<0)&(y<0)
        mask4 = (x==0)&(y>0)
        mask5 = (x==0)&(y<0)

        phi = torch.where(mask1,torch.atan(y/x),phi)
        phi = torch.where(mask2,torch.atan(y/x) + math.pi,phi)
        phi = torch.where(mask3,torch.atan(y/x) - math.pi,phi)
        phi = torch.where(mask4,math.pi/2,phi)
        phi = torch.where(mask5,-math.pi/2,phi)

        return phi

    # def Compute_xyz_in_out(self):

    #     r'''
    #     Compute the xy position where the muon enters/exits the Volume of Interest.
    #     Computation assumes that the detector have the same number of planes
    #     abopve and below the VOI.

    #     !!! WARNING currently hardcoded for n_plane = 3 !!!
    #     '''

    #     xyz_in = torch.zeros((3,2,self.n_event))
    #     xyz_out = torch.zeros((3,2,self.n_event))

    #     dz = (abs(self.hits[2,2] - self.voi.xyz_min[2]),abs(self.hits[2,3] - self.voi.xyz_max[2]))
    #     xyz_in[0] = self.hits[0,2] + dz[0]*torch.tan(self.theta_x_in)
    #     xyz_in[1] = self.hits[1,2] + dz[0]*torch.tan(self.theta_y_in)
    #     xyz_in[2] = torch.ones((self.n_event))*self.voi.xyz_min[2]

    #     xyz_out[0] = self.hits[0,3] - dz[1]*torch.tan(self.theta_x_out)
    #     xyz_out[1] = self.hits[1,3] - dz[1]*torch.tan(self.theta_y_out)
    #     xyz_out[2] = torch.ones((self.n_event))*self.voi.xyz_max[2]

    #     return xyz_in, xyz_out
    
    def Compute_xyz_in_out_new(self) -> Tuple[Tensor]:

        r"""
        Compute muon position (x,y,z) when enters/exits the VOI,
        both for the incoming and outgoing tracks.

        MUST BE CHANGED!! Must take into account the point provided
        by the fit as a reference for track computation and not the 
        last hit close to the VOI!!!
        
        INPUT:
        - hits [3,n_plane,n_event]
        - VOI.xyz_min, VOI.xyz_max
        
        OUTPUT:
        - xyz_in_VOI [3,2,n_event] the muon position when entering/exiting the VOI,
        for the INCOMING track
        
        - xyz_out_VOI [3,2,n_event] the muon position when entering/exiting the VOI,
        for the OUTGOING track
        """

        # indices of planes directly above and below the voi
        i_plane_above = int(self.n_plane/2)-1
        i_plane_below = int(self.n_plane/2)

        thetas_in = [self.theta_x_in,self.theta_y_in]
        thetas_out = [self.theta_x_out,self.theta_y_out]

        xyz_in_VOI,xyz_out_VOI = torch.zeros((3,2,len(self.theta_x_in))),torch.zeros((3,2,len(self.theta_x_out)))
            
        for i_plane,theta_xy,pm,xyz in zip([i_plane_above,i_plane_below],
                                        [thetas_in,thetas_out],
                                        [1,-1],
                                        [xyz_in_VOI,xyz_out_VOI]):
            
            dz = (abs(self.hits[2,i_plane]-self.voi.xyz_max[2]),
                abs(self.hits[2,i_plane]-self.voi.xyz_min[2]))
            
            for coord,theta in zip([0,1],theta_xy):
            
                xyz[coord,0] = self.hits[coord,i_plane] + dz[1]*torch.tan(theta)*(pm)
                xyz[coord,1] = self.hits[coord,i_plane] + dz[0]*torch.tan(theta)*(pm)
            
            xyz[2,0],xyz[2,1] = self.voi.xyz_min[2],self.voi.xyz_max[2]

        return xyz_in_VOI,xyz_out_VOI

    def Compute_xyz_in_new(self) -> Tuple[Tensor]:

        r"""
        Compute muon position (x,y,z) when enters/exits the VOI,
        both for the incoming tracks.

        MUST BE CHANGED!! Must take into account the point provided
        by the fit as a reference for track computation and not the 
        last hit close to the VOI!!!
        
        INPUT:
        - hits [3,n_plane,n_event]
        - VOI.xyz_min, VOI.xyz_max
        
        OUTPUT:
        - xyz_in_VOI [3,2,n_event] the muon position when entering/exiting the VOI,
        for the INCOMING track
        
        - xyz_out_VOI [3,2,n_event] the muon position when entering/exiting the VOI,
        for the OUTGOING track
        """

        theta = [self.theta_x,self.theta_y]

        xyz_in_VOI = torch.zeros((3,2,len(self.theta_x)))
   
        dz = (abs(self.hits[2,0]-self.voi.xyz_max[2]),
            abs(self.hits[2,0]-self.voi.xyz_min[2]))
            
        for coord,theta in zip([0,1],theta):
        
            xyz_in_VOI[coord,0] = self.hits[coord,0] - dz[1]*torch.tan(theta)
            xyz_in_VOI[coord,1] = self.hits[coord,0] - dz[0]*torch.tan(theta)
            
        xyz_in_VOI[2,0],xyz_in_VOI[2,1] = self.voi.xyz_min[2],self.voi.xyz_max[2]

        return xyz_in_VOI

    
    def Compute_discrete_tracks_scat(self,n_point_per_z_layer:int=3) -> Tuple[Tensor]:

        r"""
        This method discretize the muon track within the VOI. Will be used by ASR-like reconstruction algorithms.

        Compute muon position (x,y,z) when inside the VOI. A muon track is defined as X points,
        where X = n_layer_z x n_point_per_z_layer.

        n_point_per_z_layer must be not too small (all the voxels are not triggered)
        nor too large (computationaly expensive). A better implementation should be provided, 
        taking into account the size in [m] of the VOI as well as the voxels size.
        
        INPUT:
        - xyz_in_VOI,xyz_out_VOI [3,2,n_event] 
        the muon position when entering/exiting the VOI for incoming (xyz_in_VOI)
        and outgoing (xyz_out_VOI) tracks


        OUTPUT:
        - xyz_discrete_in, xyz_discrete_out muon position when enters/exits 
        a layer of voxels along z for incoming and outgoing tracks, with size [3,X,n_event] 
        """
        n_points = (self.voi.n_vox_xyz[2]+1)*n_point_per_z_layer
        
        z_discrete = (torch.linspace(torch.min(self.voi.voxel_edges[0,0,:,:,2]),
                                     torch.max(self.voi.voxel_edges[0,0,:,:,2]),
                                     n_points)[:,None]).expand(-1,len(self.dtheta))

        xyz_discrete_in = torch.ones((3,n_points,len(self.dtheta)))
        xyz_discrete_out = torch.ones_like(xyz_discrete_in)

        xyz_discrete_in[0] = abs(z_discrete-self.xyz_in_VOI[2,0])*torch.tan(self.theta_x_in)+self.xyz_in_VOI[0,0]
        xyz_discrete_in[1] = abs(z_discrete-self.xyz_in_VOI[2,0])*torch.tan(self.theta_y_in)+self.xyz_in_VOI[1,0]

        xyz_discrete_out[0] = abs(z_discrete-self.xyz_out_VOI[2,0])*torch.tan(self.theta_x_out)+self.xyz_out_VOI[0,0]
        xyz_discrete_out[1] = abs(z_discrete-self.xyz_out_VOI[2,0])*torch.tan(self.theta_y_out)+self.xyz_out_VOI[1,0]

        xyz_discrete_in[2], xyz_discrete_out[2] = z_discrete,z_discrete

        return xyz_discrete_in,xyz_discrete_out
    
    def Compute_discrete_tracks_abs(self,n_point_per_z_layer:int=3) -> Tuple[Tensor]:

        r"""
        This method discretize the muon track within the VOI. Will be used by ASR-like reconstruction algorithms.

        Compute muon position (x,y,z) when inside the VOI. A muon track is defined as X points,
        where X = n_layer_z x n_point_per_z_layer.

        n_point_per_z_layer must be not too small (all the voxels are not triggered)
        nor too large (computationaly expensive). A better implementation should be provided, 
        taking into account the size in [m] of the VOI as well as the voxels size.
        
        INPUT:
        - xyz_in_VOI [3,2,n_event] 
        the muon position when entering/exiting the VOI for incoming (xyz_in_VOI) tracks.

        OUTPUT:
        - xyz_discrete_in, muon position in the VOI with size [3,X,n_event] 
        """
        n_points = (self.voi.n_vox_xyz[2]+1)*n_point_per_z_layer
        
        z_discrete = (torch.linspace(torch.min(self.voi.voxel_edges[0,0,:,:,2]),
                                     torch.max(self.voi.voxel_edges[0,0,:,:,2]),
                                     n_points)[:,None]).expand(-1,self.n_event)

        xyz_discrete_in = torch.ones((3,n_points,self.n_event))

        xyz_discrete_in[0] = abs(z_discrete-self.xyz_in_VOI[2,0])*torch.tan(self.theta_x)+self.xyz_in_VOI[0,0]
        xyz_discrete_in[1] = abs(z_discrete-self.xyz_in_VOI[2,0])*torch.tan(self.theta_y)+self.xyz_in_VOI[1,0]
        xyz_discrete_in[2] = z_discrete

        return xyz_discrete_in

    def Apply_mask(self,mask:Tensor)->None:
        
        for key in self.__dict__.keys():
            attribute = getattr(self,key)
            if(type(attribute)==torch.Tensor):
                if((len(attribute.size())==1)|(len(attribute.size())==2)):
                    setattr(self,key,attribute[mask])
                if(len(attribute.size())==3):
                    setattr(self,key,attribute[:,:,mask])

        # Events
        self.n_event_masked = len(self.events)

    def save(self,filename:str) -> None:

        r"""
        Save the class on a binary file.

        WARNING: 
        The pickle module is not secure. Only unpickle data you trust.
        It is possible to construct malicious pickle data which will execute arbitrary code during unpickling. 
        Never unpickle data that could have come from an untrusted source, or that could have been tampered with.
        """

        with open(filename, 'wb') as f:
            pickle.dump(self,f)
        





    





