import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from typing import Dict, List, Union, Tuple, Optional

from tracking.Tracking import *
from volume.Volume_Interest import *
from reconstruction.ASR import ASR


from IPython.display import display, Math

# Scattering density predictions
def plot_VOI_pred(preds:Tensor, 
                  true:Tensor=None,
                  filename:str=None,
                  reverse:bool=False)->None:

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    scoresXZ = torch.transpose(torch.sum(preds,dim=1)/preds.size()[1],0,1)
    scoresXY = torch.transpose(torch.sum(preds,dim=2)/preds.size()[2],0,1)
    scoresYZ = torch.transpose(torch.sum(preds,dim=0)/preds.size()[0],0,1)
    
    if(reverse):
        cmap = "viridis_r"
    else:
        cmap = "viridis"

    fig1,ax=plt.subplots(nrows=1, ncols=1,figsize=(6.6,5))
    ax.set_title('XZ view')
    ax.set_xlabel('x [a.u]')
    ax.set_ylabel('z [a.u]')
    im = ax.imshow(scoresXZ,cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig1.colorbar(im, ax=ax,cax=cax)
    plt.tight_layout()
    if(filename is not None):
        plt.savefig(filename+'_XZ_view')
    plt.show()

    fig1,ax=plt.subplots(nrows=1, ncols=1,figsize=(6.6,5))
    ax.set_title('XY view')
    ax.set_xlabel('x [a.u]')
    ax.set_ylabel('y [a.u]')
    im = ax.imshow(scoresXY,cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig1.colorbar(im, ax=ax,cax=cax)
    plt.tight_layout()
    if(filename is not None):
        plt.savefig(filename+'_XY_view')
    plt.show()
    
    fig1,ax=plt.subplots(nrows=1, ncols=1,figsize=(6.6,5))
    ax.set_title('YZ view')
    ax.set_xlabel('y [a.u]')
    ax.set_ylabel('z [a.u]')
    im = ax.imshow(scoresYZ,cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig1.colorbar(im, ax=ax,cax=cax)
    plt.tight_layout()
    if(filename is not None):
        plt.savefig(filename+'_YZ_view')
    plt.show()


# POCA points location
def plot_POCA_points(POCA_points:Tensor, 
                     mask:Tensor=None, 
                     binning_xyz:Tuple[int]=[100,100,100],
                     filename:str =None) -> None:

    if(mask is None):
        mask = np.ones(POCA_points.size()[0],dtype=bool)
    p = POCA_points.numpy()
    
    scale = 2.
    # XY view
    fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(scale*10,scale*5.5))
    ax = ax.ravel()

    ax[0].hist(p[:,0][mask],bins=binning_xyz[0])
    ax[0].set_xlabel('Voxels along x')
    ax[0].set_ylabel('# POCA points per voxel')


    ax[1].hist(p[:,1][mask],bins=binning_xyz[1])
    ax[1].set_xlabel('Voxels along y')
    ax[0].set_ylabel('# POCA points per voxel')


    ax[2].hist(p[:,2][mask],bins=binning_xyz[2])
    ax[2].set_xlabel('Voxels along z')
    ax[0].set_ylabel('# POCA points per voxel')


    ax[3].hist2d(p[:,0],p[:,1],bins=(binning_xyz[0],binning_xyz[1]))
    ax[3].set_aspect('equal')
    ax[3].set_xlabel('x [mm]')
    ax[3].set_ylabel('y [mm]')


    ax[4].hist2d(p[:,0],p[:,2],bins=(binning_xyz[0],binning_xyz[2]))
    ax[4].set_aspect('equal')
    ax[4].set_xlabel('x [mm]')
    ax[4].set_ylabel('z [mm]')

    ax[5].hist2d(p[:,1],p[:,2],bins=(binning_xyz[1],binning_xyz[2]))
    ax[5].set_aspect('equal')
    ax[5].set_xlabel('y [mm]')
    ax[5].set_ylabel('z [mm]')

    plt.tight_layout()
    if(filename is not None):
        plt.savefig(filename+"_XY_view")
    plt.show()


def plot_POCA_points_multi_projection(POCA_points:Tensor, 
                                      mask:Tensor=None, 
                                      binning_xyz:Tuple[int]=[100,100,100],
                                      filename:str =None) -> None:

    if(mask is None):
        mask = np.ones(POCA_points.size()[0],dtype=bool)
    p = POCA_points.numpy()
    
    # XY view
    fig,main_ax = plt.subplots(figsize=(8, 8))

    # 2D hits on the main axes
    main_ax.hist2d(p[:,0][mask],p[:,1][mask],bins=(binning_xyz[0],binning_xyz[1]))
    main_ax.set_aspect('equal')
    main_ax.set_xlabel('x [mm]',fontsize=14)
    main_ax.set_ylabel('y [mm]',fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    # X view
    x_hist = main_ax.inset_axes([0, 1.20, 1, .5], transform=main_ax.transAxes, sharex=main_ax)

    x_hist.hist(p[:,0][mask],bins=binning_xyz[0], histtype='stepfilled',
                orientation='vertical',alpha=.7)
    x_hist.set_ylabel('# POCA points per voxel',fontsize=14)
    x_hist.tick_params(axis='x', labelsize=14)
    x_hist.tick_params(axis='y', labelsize=14)

    # Y view
    y_hist = main_ax.inset_axes([1.5, 0, .5, 1], transform=main_ax.transAxes,sharey=main_ax)

    y_hist.hist(p[:,1][mask], bins=binning_xyz[1],
                orientation='horizontal',alpha=.7)
    y_hist.set_xlabel('# POCA points per voxel',fontsize=14)
    y_hist.tick_params(axis='x', labelsize=14)
    y_hist.tick_params(axis='y', labelsize=14)

    if(filename is not None):
        plt.savefig(filename+'_XY_view',bbox_inches = 'tight')
    plt.show()
    
    # XZ view
    fig,main_ax = plt.subplots(figsize=(8, 8))

    # 2D hits on the main axes
    main_ax.hist2d(p[:,0][mask],-p[:,2][mask],bins=(binning_xyz[0],binning_xyz[2]))
    main_ax.set_aspect('equal')
    main_ax.set_xlabel('x [mm]',fontsize=14)
    main_ax.set_ylabel('z [mm]',fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    # X view
    x_hist = main_ax.inset_axes([0, 1.20, 1, .5], transform=main_ax.transAxes, sharex=main_ax)

    x_hist.hist(p[:,0][mask],bins=binning_xyz[0], histtype='stepfilled',
                orientation='vertical',alpha=.7)
    x_hist.set_ylabel('# POCA points per voxel',fontsize=14)
    x_hist.tick_params(axis='x', labelsize=14)
    x_hist.tick_params(axis='y', labelsize=14)

    # Y view
    y_hist = main_ax.inset_axes([1.5, 0, .5, 1], transform=main_ax.transAxes,sharey=main_ax)

    y_hist.hist(-p[:,2][mask], bins=binning_xyz[2],
                orientation='horizontal',alpha=.7)
    y_hist.set_xlabel('# POCA points per voxel',fontsize=14)
    y_hist.tick_params(axis='x', labelsize=14)
    y_hist.tick_params(axis='y', labelsize=14)

    if(filename is not None):
        plt.savefig(filename+'_XZ_view',bbox_inches = 'tight')
    plt.show()

    # YZ view
    fig,main_ax = plt.subplots(figsize=(8, 8))

    # 2D hits on the main axes
    main_ax.hist2d(p[:,1][mask],-p[:,2][mask],bins=(binning_xyz[1],binning_xyz[2]))
    main_ax.set_aspect('equal')
    main_ax.set_xlabel('y [mm]',fontsize=14)
    main_ax.set_ylabel('z [mm]',fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    # X view
    x_hist = main_ax.inset_axes([0, 1.20, 1, .5], transform=main_ax.transAxes, sharex=main_ax)

    x_hist.hist(p[:,1][mask],bins=binning_xyz[1], histtype='stepfilled',
                orientation='vertical',alpha=.7)
    x_hist.set_ylabel('# POCA points per voxel',fontsize=14)
    x_hist.tick_params(axis='x', labelsize=14)
    x_hist.tick_params(axis='y', labelsize=14)

    # Y view
    y_hist = main_ax.inset_axes([1.2, 0, .5, 1], transform=main_ax.transAxes, sharey=main_ax)
    # y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)

    y_hist.hist(-p[:,2][mask], bins=binning_xyz[2],
                orientation='horizontal',alpha=.7)
    y_hist.set_xlabel('# POCA points per voxel',fontsize=14)
    y_hist.tick_params(axis='x', labelsize=14)
    y_hist.tick_params(axis='y', labelsize=14)

    if(filename is not None):
        plt.savefig(filename+'_YZ_view',bbox_inches = 'tight')
    plt.show()

# POCA points location by slice
def get_fig_size(VOI:VolumeInterest,
                 nrows:int,
                 dims:Tuple[int]=(0,1),
                 ncols:int=3,
                 scale:float=3.) -> Tuple[float]:
    
    dx, dy = VOI.dxyz[dims[0]], VOI.dxyz[dims[1]]
    
    if dx>dy: 
        xy_ratio=[1.,float(dy/dx)]
    else: 
        xy_ratio=[float(dx/dy),1.]
        
    return (scale*ncols*(xy_ratio[0]+1/4),scale*nrows*(xy_ratio[1]+1/4))


def get_nrows(nplots:int,
              ncols:int) -> Tuple[int]:
    
    if nplots%ncols==0:
        nrows, extra=int(nplots/ncols),0
    else:
        for i in range(1,ncols):
            if nplots%ncols==i: nrows, extra=int(nplots/ncols)+1,ncols-i
    return nrows, extra


def get_vmin_vmax(VOI:VolumeInterest, 
                  poca_points, 
                  dim:int,
                  scale_binning:float=1.5) -> Tuple[float]:
    
    mins,maxs = [],[]
    
    for i,z_min in enumerate(range(int(VOI.xyz_min[dim]), 
                                   int(VOI.xyz_max[dim]), 
                                   int(VOI.vox_width))):
        
        z_max = z_min + VOI.vox_width 
        mask_slice = (poca_points[:,dim]>=z_min)&((poca_points[:,dim]<=z_max))
        
        if dim==2: dim_xy=(0,1)
        elif dim==1: dim_xy=(0,2)
        else: dim_xy=(1,2)
        
        # get xy binning
        bins = (int(scale_binning*VOI.n_vox_xyz[dim_xy[0]]),int(scale_binning*VOI.n_vox_xyz[dim_xy[1]]))
        
        # compute 2D histogram
        H, xedges, yedges = np.histogram2d(poca_points[mask_slice,dim_xy[0]], 
                                           poca_points[mask_slice,dim_xy[1]], 
                                           bins=bins)
        
        # get min and max of the current slice
        mins.append(np.min(H))
        maxs.append(np.max(H))
        
    return np.min(np.array(mins)),np.max(np.array(maxs))


def plot_poca_points_by_voi_slice(poca_points:torch.tensor,
                                  VOI:VolumeInterest, 
                                  dim:int=2,
                                  ncols:int=4,
                                  scale_binning:float=1.,
                                  filename:str=None,
                                  axis_label_all:bool=False,
                                  fontsize:int=17) -> None:
    
    """
    Plots POCA points location as 2D histogram, for each slice of the VOI along a given dimension.
    
    INPUT:
    
    - poca_points:torch.tensor with size (n,3) containing the 
    locations in x,y,z of the n poca points.
    - VOI:VolumeInterest, an instance of the volume of interest.
    - dim:int=2, the dimension to consider. dim=0: x, dim=1: y, dim=2: z. 
    e.g, if dim=2 poca points are represented as a 2D histogram in x and y 
    for each VOI slice along z.
    - ncols:int=4, the number of columns of the figure.
    - scale_binning:float=1., the rescaling of the histogram binning w.r.t
    the VOI initial voxeliztion. By default, 1 histogram bin = 1 voxel. If scale_binning=2, 
    1 histogram bin = 2 voxels.
    - filename:str=None, if filename is not None, figure save the figure as "filename".
    - axis_label_all:bool=False. If True, axis labels are shown on every subplot.
    """
    # set fontsize
    font = {'weight' : 'bold',
            'size'   : fontsize}
    
    import matplotlib
    matplotlib.rc('font', **font)
    
    nplots = int(VOI.n_vox_xyz[dim])
    
    if dim==2: dim_xy, dim_label=(0,1), ["x","y","z"]
    elif dim==1: dim_xy, dim_label=(0,2), ["x","z","y"]
    else: dim_xy, dim_label=(1,2), ["y","z","x"]
    
    # compute nrows given ncols
    # extra the number of blank plots (if nplots%ncols!=0)
    nrows, extra = get_nrows(nplots=nplots,
                             ncols=ncols)
        
    # compute figure size
    figsize = get_fig_size(VOI,
                           nrows=nrows,
                           ncols=ncols,
                           dims=dim_xy,
                           scale=4)

    fig, axs = plt.subplots(ncols=ncols,
                            nrows=nrows,
                            figsize=figsize,
                            sharex=True,
                            sharey=True)
    # get xy binning
    bins = (int(scale_binning*VOI.n_vox_xyz[dim_xy[0]]),
            int(scale_binning*VOI.n_vox_xyz[dim_xy[1]]))
    
    axs = axs.ravel()
    pixel_size = VOI.dxyz[dim_xy[0]]/bins[0]
    
    # fig title
    fig.suptitle("POCA points 2D distribution\n # POCA points = {}, pixel size = {} mm".format(int(poca_points.size()[0]),pixel_size),y=1.0)

    # compute min and max value for colorscale
    vmin,vmax=get_vmin_vmax(VOI=VOI,
                            poca_points=poca_points,
                            dim=dim,
                            scale_binning=scale_binning)
    
    # get the VOI min and max XY position
    xmin,xmax = VOI.xyz_min[dim_xy[0]],VOI.xyz_max[dim_xy[0]]
    ymin,ymax = VOI.xyz_min[dim_xy[1]],VOI.xyz_max[dim_xy[1]]
    
    # Loop ober the number of voxels along `dim` dimension
    for i,z_min in enumerate(range(int(VOI.xyz_min[dim]), 
                                   int(VOI.xyz_max[dim]), 
                                   int(VOI.vox_width))):
        
        # get poca points within the slice
        z_max = z_min + VOI.vox_width 
        mask_slice = (poca_points[:,dim]>=z_min)&((poca_points[:,dim]<=z_max))
        
        

        # compute 2D histogram
        H, xedges, yedges = np.histogram2d(poca_points[mask_slice,dim_xy[0]], 
                                           poca_points[mask_slice,dim_xy[1]], 
                                           bins=bins,
                                           range=((xmin,xmax),(ymin,ymax)))
        im = axs[i].imshow(H.T, 
                           interpolation='nearest', 
                           origin='lower',
                           extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],
                           vmin=vmin,
                           vmax=vmax)

        axs[i].set_title(r"{} $\in$ [{},{}] mm".format(dim_label[2],z_min,z_max))
        axs[i].set_aspect("equal")
        
        # set axes x label 
        if (i%ncols==0) | (i==0) | axis_label_all:
            axs[i].set_ylabel("{} [mm]".format(dim_label[1]))
        # set axes y label    
        if (i>=nplots-ncols) | axis_label_all:
            axs[i].set_xlabel("{} [mm]".format(dim_label[0]))
    
    # remove empty axis 
    for i in range(1,extra+1):
        axs[-i].remove()
        axs[-i] = None
        
    # Add color bar
    cbar_ax = fig.add_axes([1.01, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax,label="# POCA points per pixel")
        
    # make it look nicer
    plt.tight_layout()
    
    # save
    if filename is not None: plt.savefig(filename)
    plt.show()

    
def plot_event_reconstruction(tracks:Tracking,
                              event:int=None,
                              asr:ASR = None,
                             hits=False)->None:
    VOI = tracks.voi
    if event is None:
        event = np.random.randint(tracks.n_event)
    
    def plot_hits(ax,dim0:int=0,dim1:int=2):
        ax.scatter(tracks.hits[dim0,:int(tracks.n_plane/2),event],
                tracks.hits[dim1,:int(tracks.n_plane/2),event], 
                color = 'green', 
                label='track in',
                marker = '^',
                s=150)

        ax.scatter(tracks.hits[dim0,int(tracks.n_plane/2):,event],
                tracks.hits[dim1,int(tracks.n_plane/2):,event], 
                color = 'red', 
                label='track out',
                marker = 'v',
                s=200)
    
    def plot_in_out_VOI(ax,dim0:int=0,dim1:int=2):
        
        ax.scatter(tracks.xyz_in_VOI[dim0,:,event],
                tracks.xyz_in_VOI[dim1,:,event],
                color='green',
                label = 'track in',
                marker = 'x',
                s=150)
        
        ax.scatter(tracks.xyz_out_VOI[dim0,:,event],
                tracks.xyz_out_VOI[dim1,:,event],
                color='red',
                label = 'track out',
                marker = '+',
                s=200)
    
    def set_ax_xy_lim(ax,x_values=None,y_values=None,gap:float=200.):
        if(x_values is not None):
            x_min = torch.min((x_values))
            x_max = torch.max((x_values))
            ax.set_xlim((x_min-gap,x_max+gap))
            
        if(y_values is not None):
            y_min = torch.min((y_values))
            y_max = torch.max((y_values))
            ax.set_ylim((y_min-gap,y_max+gap))
        
        
    def plot_voxel_grid(ax,proj:str='XZ'):
        xmin,xmax,ymin,ymax = 0,1,0,1
        if((proj=='XZ')|(proj=='ZX')):
            
            x_width = abs(ax.get_xlim()[0]-ax.get_xlim()[1])
            y_width = abs(ax.get_ylim()[0]-ax.get_ylim()[1])
            
            if(ax.get_xlim()[0]<VOI.xyz_min[0]):
                xmin = np.abs(ax.get_xlim()[0]-VOI.xyz_min[0])/x_width
                
            if(ax.get_xlim()[1]>VOI.xyz_max[0]):
                xmax = 1. - np.abs(ax.get_xlim()[1]-VOI.xyz_max[0])/x_width
            
            if(ax.get_ylim()[0]<VOI.xyz_min[2]):
                ymin = np.abs(ax.get_ylim()[0]-VOI.xyz_min[2])/y_width
            
            if(ax.get_ylim()[1]>VOI.xyz_max[2]):
                ymax = 1. - np.abs(ax.get_ylim()[1]-VOI.xyz_max[2])/y_width
            
            for i in range(VOI.n_vox_xyz[2]):
                ax.axhline(VOI.voxel_edges[0,0,i,0,2],color='blue',alpha=.3,xmin=float(xmin),xmax=float(xmax))
                ax.axhline(VOI.voxel_edges[0,0,i,1,2],color='blue',alpha=.3,xmin=float(xmin),xmax=float(xmax))
        
            for i in range(VOI.n_vox_xyz[0]):
                ax.axvline(VOI.voxel_edges[i,0,0,0,0],color='blue',alpha=.3,ymin=float(ymin),ymax=float(ymax))
                ax.axvline(VOI.voxel_edges[i,0,0,1,0],color='blue',alpha=.3,ymin=float(ymin),ymax=float(ymax))
                
        if((proj=='YZ')|(proj=='ZY')):
            
            x_width = abs(ax.get_xlim()[0]-ax.get_xlim()[1])
            y_width = abs(ax.get_ylim()[0]-ax.get_ylim()[1])
            
            if(ax.get_xlim()[0]<VOI.xyz_min[1]):
                xmin = abs(ax.get_xlim()[0]-VOI.xyz_min[1])/x_width
                
            if(ax.get_xlim()[1]>VOI.xyz_max[1]):
                xmax = 1 - abs(ax.get_xlim()[1]-VOI.xyz_max[1])/x_width
            
            if(ax.get_ylim()[0]<VOI.xyz_min[2]):
                ymin = abs(ax.get_ylim()[0]-VOI.xyz_min[2])/y_width
            
            if(ax.get_ylim()[1]>VOI.xyz_max[2]):
                ymax = 1 - abs(ax.get_ylim()[1]-VOI.xyz_max[2])/y_width
            
            for i in range(VOI.n_vox_xyz[2]):
                ax.axhline(VOI.voxel_edges[0,0,i,0,2],color='blue',alpha=.3,xmin=float(xmin),xmax=float(xmax))
                ax.axhline(VOI.voxel_edges[0,0,i,1,2],color='blue',alpha=.3,xmin=float(xmin),xmax=float(xmax))
        
            for i in range(VOI.n_vox_xyz[1]):
                ax.axvline(VOI.voxel_edges[0,i,0,0,1],color='blue',alpha=.3,ymin=float(ymin),ymax=float(ymax))
                ax.axvline(VOI.voxel_edges[0,i,0,1,1],color='blue',alpha=.3,ymin=float(ymin),ymax=float(ymax))
                
                
        if((proj=='XY')|(proj=='YX')):
            
            x_width = abs(ax.get_xlim()[0]-ax.get_xlim()[1])
            y_width = abs(ax.get_ylim()[0]-ax.get_ylim()[1])
            
            if(ax.get_xlim()[0]<VOI.xyz_min[0]):
                xmin = abs(ax.get_xlim()[0]-VOI.xyz_min[0])/x_width
                
            if(ax.get_xlim()[1]>VOI.xyz_max[0]):
                xmax = 1 - abs(ax.get_xlim()[1]-VOI.xyz_max[0])/x_width
            
            if(ax.get_ylim()[0]<VOI.xyz_min[1]):
                ymin = abs(ax.get_ylim()[0]-VOI.xyz_min[1])/y_width
                
            if(ax.get_ylim()[1]>VOI.xyz_max[1]):
                ymax = 1 - abs(ax.get_ylim()[1]-VOI.xyz_max[1])/y_width
            
            for i in range(VOI.n_vox_xyz[1]):
                ax.axhline(VOI.voxel_edges[0,i,0,0,1],color='blue',alpha=.3,xmin=float(xmin),xmax=float(xmax))
                ax.axhline(VOI.voxel_edges[0,i,0,1,1],color='blue',alpha=.3,xmin=float(xmin),xmax=float(xmax))
        
            for i in range(VOI.n_vox_xyz[0]):
                ax.axvline(VOI.voxel_edges[i,1,0,0,0],color='blue',alpha=.3,ymin=float(ymin),ymax=float(ymax))
                ax.axvline(VOI.voxel_edges[i,1,0,1,0],color='blue',alpha=.3,ymin=float(ymin),ymax=float(ymax))
       
    
    def plot_triggered_vox(ax):    
        
        vox_indices = asr.triggered_voxels[event]
        

        if(len(vox_indices)>=1):
            ix = vox_indices[:,0]
            iy = vox_indices[:,1]
            iz = vox_indices[:,2]

            ax[0].scatter(VOI.voxel_centers[ix,iy,iz,0],
                        VOI.voxel_centers[ix,iy,iz,2],
                        c='grey',
                        alpha=.3)
            ax[1].scatter(VOI.voxel_centers[ix,iy,iz,1],
                        VOI.voxel_centers[ix,iy,iz,2],
                        c='grey',
                        alpha=.3)
            ax[2].scatter(VOI.voxel_centers[ix,iy,iz,0],
                        VOI.voxel_centers[ix,iy,iz,1],
                        c='grey',
                        alpha=.3)
    
    fig,ax = plt.subplots(ncols=3,figsize=(15,5))
        
    # Plot hits
    if(hits):
        plot_hits(ax[0])
        plot_hits(ax[1],dim0=1)
        plot_hits(ax[2],dim0=0,dim1=1)

    # Plot muon in/out VOI
    plot_in_out_VOI(ax[0])
    plot_in_out_VOI(ax[1],dim0=1)
    plot_in_out_VOI(ax[2],dim1=1)

    # Plot triggered voxels
    if(asr is not None):
        plot_triggered_vox(ax)

    # Axes setting
    # Range
    if(hits):
        set_ax_xy_lim(ax[0],x_values=torch.cat((tracks.hits[0,:,event],
                                                tracks.hits[0,:,event])))
        set_ax_xy_lim(ax[1],x_values=torch.cat((tracks.hits[1,:,event],
                                                tracks.hits[1,:,event])))
        set_ax_xy_lim(ax[2],x_values=torch.cat((tracks.hits[0,:,event],
                                                tracks.hits[0,:,event])),
                            y_values=torch.cat((tracks.hits[1,:,event],
                                                tracks.hits[1,:,event])))
        
    else:
        set_ax_xy_lim(ax[0],x_values=torch.cat((tracks.xyz_in_VOI[0,:,event],
                                                tracks.xyz_out_VOI[0,:,event])))
        set_ax_xy_lim(ax[1],x_values=torch.cat((tracks.xyz_in_VOI[1,:,event],
                                                tracks.xyz_out_VOI[1,:,event])))
        set_ax_xy_lim(ax[2],x_values=torch.cat((tracks.xyz_in_VOI[0,:,event],
                                                tracks.xyz_out_VOI[0,:,event])),
                            y_values=torch.cat((tracks.xyz_in_VOI[1,:,event],
                                                tracks.xyz_out_VOI[1,:,event])))
       
    # Labels
    ax[0].set_aspect('equal')
    ax[0].set_xlabel('x [mm]')
    ax[0].set_ylabel('z [mm]')

    ax[1].set_aspect('equal')
    ax[1].set_xlabel('y [mm]')
    ax[1].set_ylabel('z [mm]')

    ax[2].set_aspect('equal')
    ax[2].set_xlabel('x [mm]')
    ax[2].set_ylabel('y [mm]')

    # plot voxel grid
    plot_voxel_grid(ax[0],proj='XZ')
    plot_voxel_grid(ax[1],proj='YZ')
    plot_voxel_grid(ax[2],proj='XY')

    # fig
    if(asr is None):
        fig.suptitle("Tracking reconstruction for event {}\nScattering angle = {:.3f} deg".format(event,tracks.dtheta[event]))    
    else:
        fig.suptitle("Tracking reconstruction for event {}\nScattering angle = {:.3f} deg\n # triggered voxels = {}".format(event,tracks.dtheta[event],len(asr.triggered_voxels[event])))    

    fig.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.show()

    
def plot_event_reconstruction_abs(tracks:Tracking,
                                  event:int=None,
                                  asr:ASR = None,
                                  hits=False) -> None:
        
    VOI = tracks.voi
    if event is None:
        event = np.random.randint(tracks.n_event)
    
    def plot_hits(ax,dim0:int=0,dim1:int=2):
        ax.scatter(tracks.hits[dim0,:,event],
                tracks.hits[dim1,:,event], 
                color = 'green', 
                label='track',
                marker = '^',
                s=150)

    def plot_in_out_VOI(ax,dim0:int=0,dim1:int=2):
        
        ax.scatter(tracks.xyz_in_VOI[dim0,:,event],
                tracks.xyz_in_VOI[dim1,:,event],
                color='green',
                label = 'track',
                marker = 'x',
                s=150)
        
    
    def set_ax_xy_lim(ax,x_values=None,y_values=None,gap:float=200.):
        if(x_values is not None):
            x_min = torch.min((x_values))
            x_max = torch.max((x_values))
            ax.set_xlim((x_min-gap,x_max+gap))
            
        if(y_values is not None):
            y_min = torch.min((y_values))
            y_max = torch.max((y_values))
            ax.set_ylim((y_min-gap,y_max+gap))
        
        
    def plot_voxel_grid(ax,proj:str='XZ'):
        xmin,xmax,ymin,ymax = 0,1,0,1
        if((proj=='XZ')|(proj=='ZX')):
            
            x_width = abs(ax.get_xlim()[0]-ax.get_xlim()[1])
            y_width = abs(ax.get_ylim()[0]-ax.get_ylim()[1])
            
            if(ax.get_xlim()[0]<VOI.xyz_min[0]):
                xmin = float(abs(ax.get_xlim()[0]-VOI.xyz_min[0])/x_width)
                
            if(ax.get_xlim()[1]>VOI.xyz_max[0]):
                xmax = float(1 - abs(ax.get_xlim()[1]-VOI.xyz_max[0])/x_width)
            
            if(ax.get_ylim()[0]<VOI.xyz_min[2]):
                ymin = float(abs(ax.get_ylim()[0]-VOI.xyz_min[2])/y_width)
            
            if(ax.get_ylim()[1]>VOI.xyz_max[2]):
                ymax = float(1 - abs(ax.get_ylim()[1]-VOI.xyz_max[2])/y_width)
            
            for i in range(VOI.n_vox_xyz[2]):
                ax.axhline(VOI.voxel_edges[0,0,i,0,2],color='blue',alpha=.3,xmin=xmin,xmax=xmax)
                ax.axhline(VOI.voxel_edges[0,0,i,1,2],color='blue',alpha=.3,xmin=xmin,xmax=xmax)
        
            for i in range(VOI.n_vox_xyz[0]):
                ax.axvline(VOI.voxel_edges[i,0,0,0,0],color='blue',alpha=.3,ymin=ymin,ymax=ymax)
                ax.axvline(VOI.voxel_edges[i,0,0,1,0],color='blue',alpha=.3,ymin=ymin,ymax=ymax)
                
        if((proj=='YZ')|(proj=='ZY')):
            
            x_width = abs(ax.get_xlim()[0]-ax.get_xlim()[1])
            y_width = abs(ax.get_ylim()[0]-ax.get_ylim()[1])
            
            if(ax.get_xlim()[0]<VOI.xyz_min[1]):
                xmin = float(abs(ax.get_xlim()[0]-VOI.xyz_min[1])/x_width)
                
            if(ax.get_xlim()[1]>VOI.xyz_max[1]):
                xmax = float(1 - abs(ax.get_xlim()[1]-VOI.xyz_max[1])/x_width)
            
            if(ax.get_ylim()[0]<VOI.xyz_min[2]):
                ymin = float(abs(ax.get_ylim()[0]-VOI.xyz_min[2])/y_width)
            
            if(ax.get_ylim()[1]>VOI.xyz_max[2]):
                ymax = float(1 - abs(ax.get_ylim()[1]-VOI.xyz_max[2])/y_width)
            
            for i in range(VOI.n_vox_xyz[2]):
                ax.axhline(VOI.voxel_edges[0,0,i,0,2],color='blue',alpha=.3,xmin=xmin,xmax=xmax)
                ax.axhline(VOI.voxel_edges[0,0,i,1,2],color='blue',alpha=.3,xmin=xmin,xmax=xmax)
        
            for i in range(VOI.n_vox_xyz[1]):
                ax.axvline(VOI.voxel_edges[0,i,0,0,1],color='blue',alpha=.3,ymin=ymin,ymax=ymax)
                ax.axvline(VOI.voxel_edges[0,i,0,1,1],color='blue',alpha=.3,ymin=ymin,ymax=ymax)
                
                
        if((proj=='XY')|(proj=='YX')):
            
            x_width = abs(ax.get_xlim()[0]-ax.get_xlim()[1])
            y_width = abs(ax.get_ylim()[0]-ax.get_ylim()[1])
            
            if(ax.get_xlim()[0]<VOI.xyz_min[0]):
                xmin = float(abs(ax.get_xlim()[0]-VOI.xyz_min[0])/x_width)
                
            if(ax.get_xlim()[1]>VOI.xyz_max[0]):
                xmax = float(1 - abs(ax.get_xlim()[1]-VOI.xyz_max[0])/x_width)
            
            if(ax.get_ylim()[0]<VOI.xyz_min[1]):
                ymin = float(abs(ax.get_ylim()[0]-VOI.xyz_min[1])/y_width)
                
            if(ax.get_ylim()[1]>VOI.xyz_max[1]):
                ymax = float(1 - abs(ax.get_ylim()[1]-VOI.xyz_max[1])/y_width)
            
            for i in range(VOI.n_vox_xyz[1]):
                ax.axhline(VOI.voxel_edges[0,i,0,0,1],color='blue',alpha=.3,xmin=xmin,xmax=xmax)
                ax.axhline(VOI.voxel_edges[0,i,0,1,1],color='blue',alpha=.3,xmin=xmin,xmax=xmax)
        
            for i in range(VOI.n_vox_xyz[0]):
                ax.axvline(VOI.voxel_edges[i,1,0,0,0],color='blue',alpha=.3,ymin=ymin,ymax=ymax)
                ax.axvline(VOI.voxel_edges[i,1,0,1,0],color='blue',alpha=.3,ymin=ymin,ymax=ymax)
                
                
    def plot_triggered_vox(ax):    
        
        vox_indices = asr.triggered_voxels[event]
        

        if(len(vox_indices)>=1):
            ix = vox_indices[:,0]
            iy = vox_indices[:,1]
            iz = vox_indices[:,2]

            ax[0].scatter(VOI.voxel_centers[ix,iy,iz,0],
                        VOI.voxel_centers[ix,iy,iz,2],
                        c='grey',
                        alpha=.3)
            ax[1].scatter(VOI.voxel_centers[ix,iy,iz,1],
                        VOI.voxel_centers[ix,iy,iz,2],
                        c='grey',
                        alpha=.3)
            ax[2].scatter(VOI.voxel_centers[ix,iy,iz,0],
                        VOI.voxel_centers[ix,iy,iz,1],
                        c='grey',
                        alpha=.3)
    
    fig,ax = plt.subplots(ncols=3,figsize=(15,5))
        
    # Plot hits
    if(hits):
        plot_hits(ax[0])
        plot_hits(ax[1],dim0=1)
        plot_hits(ax[2],dim0=0,dim1=1)

    # Plot muon in/out VOI
    plot_in_out_VOI(ax[0])
    plot_in_out_VOI(ax[1],dim0=1)
    plot_in_out_VOI(ax[2],dim1=1)

    # Plot triggered voxels
    if(asr is not None):
        plot_triggered_vox(ax)

    # Axes setting
    # Range
    if(hits):
        set_ax_xy_lim(ax[0],x_values=torch.cat((tracks.hits[0,:,event],
                                                tracks.hits[0,:,event])))
        set_ax_xy_lim(ax[1],x_values=torch.cat((tracks.hits[1,:,event],
                                                tracks.hits[1,:,event])))
        set_ax_xy_lim(ax[2],x_values=torch.cat((tracks.hits[0,:,event],
                                                tracks.hits[0,:,event])),
                            y_values=torch.cat((tracks.hits[1,:,event],
                                                tracks.hits[1,:,event])))
        
    else:
        set_ax_xy_lim(ax[0],x_values=tracks.xyz_in_VOI[0,:,event])
        set_ax_xy_lim(ax[1],x_values=tracks.xyz_in_VOI[1,:,event])
        set_ax_xy_lim(ax[2],
                      x_values=tracks.xyz_in_VOI[0,:,event],
                      y_values=tracks.xyz_in_VOI[1,:,event])
       
    # Labels
    ax[0].set_aspect('equal')
    ax[0].set_xlabel('x [mm]')
    ax[0].set_ylabel('z [mm]')

    ax[1].set_aspect('equal')
    ax[1].set_xlabel('y [mm]')
    ax[1].set_ylabel('z [mm]')

    ax[2].set_aspect('equal')
    ax[2].set_xlabel('x [mm]')
    ax[2].set_ylabel('y [mm]')

    # plot voxel grid
    plot_voxel_grid(ax[0],proj='XZ')
    plot_voxel_grid(ax[1],proj='YZ')
    plot_voxel_grid(ax[2],proj='XY')

    # fig
    if(asr is None):
        fig.suptitle("Tracking reconstruction for event {}".format(event))    
    else:
        fig.suptitle("Tracking reconstruction for event {}\n # triggered voxels = {}".format(event,len(asr.triggered_voxels[event])))    

    fig.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.show()

    
def plot_muon_batch(tracks:Tracking) -> None:

    
    if(tracks.analysis=="absorption"):
        fig,ax = plt.subplots(ncols=2,nrows=2,figsize=(15,10))
    else:
        fig,ax = plt.subplots(ncols=2,nrows=3,figsize=(15,15))

    fig.suptitle('Tracking data'+'\n# events = {}'.format(tracks.n_event))
    ax = ax.ravel()

    # Incoming zenith angle
    if(tracks.analysis=="absorption"):
        zenith = tracks.theta.numpy()*180/math.pi
    else:
        zenith = tracks.theta_in.numpy()*180/math.pi

    lab = "mean = {:.2f} deg".format(zenith.mean())
    ax[0].hist(zenith,bins=100,alpha=.5,range=(0,50),color='orange')
    ax[0].axvline(x = zenith.mean(), label = lab,color='red')
    ax[0].set_xlabel("Incoming zenith angle [deg]")
    ax[0].legend()

    # Incoming azimuthal angle
    if(tracks.analysis=="absorption"):
        azimuthal = tracks.phi.numpy()*180/math.pi
    else:
        azimuthal = tracks.phi_in.numpy()*180/math.pi
    ax[1].hist(azimuthal,bins=100,alpha=.5,color='red')
    ax[1].set_xlabel("Azymuthal angle [rad]")

    # Energy
    energy = tracks.E/1000
    lab1 = "mean = {:.2f} GeV".format(energy.mean())
    ax[2].hist(energy,bins=100,log=True,alpha=.5,range=(0,100),color='green')
    ax[2].axvline(x = energy.mean(), label = lab1,color='red')
    ax[2].set_xlabel("Energy [GeV]")
    ax[2].legend()

    # Scattering angle log
    if(tracks.analysis=="scattering"):
        dtheta = tracks.dtheta.numpy()*180/math.pi
        lab2 = "mean = {:.2f} deg\nstd = {:.2f} deg".format(dtheta.mean(),dtheta.std())
        ax[3].hist(dtheta,bins=100,log=True,alpha=.5, label = lab2)
        ax[3].set_xlabel("Scattering angle [deg]")
        ax[3].legend()

    # Scattering angle log
    if(tracks.analysis=="scattering"):
        dtheta = tracks.dtheta.numpy()*180/math.pi
        lab2 = "mean = {:.2f} deg".format(dtheta.mean())
        ax[4].hist(dtheta,bins=100,log=True,alpha=.5, range=(0,3*dtheta.mean()))
        ax[4].axvline(x = dtheta.mean(),color='red',label=lab2)
        ax[4].set_xlabel("Scattering angle [deg]")
        ax[4].legend()

    # Outgoing zenith angle
    if(tracks.analysis=="scattering"):
        zenith = tracks.theta_out.numpy()*180/math.pi
        lab = "mean = {:.2f} deg".format(zenith.mean())
        ax[5].hist(zenith,bins=100,alpha=.5,range=(0,50),color='orange')
        ax[5].axvline(x = zenith.mean(), label = lab,color='red')
        ax[5].set_xlabel("Outgoing zenith angle [deg]")
        ax[5].legend()

    plt.show()

    
def set_ticks_and_label(n_ticks:int,
                        voi_centers:torch.tensor,
                        n_vox:int,
                        ax,dim:int,fontsize:int=14):
    
    mod = int(n_vox/n_ticks)
    ticks = [i for i in range(n_vox) if (i%mod==0)]
    ticks_label = [int(pos.item()) for pos,i in (zip(voi_centers,range(n_vox))) if (i%mod==0)]
    
    if(dim==0):
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks_label,fontsize=fontsize)
    if(dim==1):
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks_label,fontsize=fontsize)

#Absorption

def imshow_transmission_rate(data:Tensor,
                             fig,
                             main_ax,
                             VOI:VolumeInterest,
                             fontsize:int=14,
                             only_imshow:bool=False,
                             z_slice:int=0):
    
    # imshow
    im = main_ax.imshow(data[:,:,z_slice],vmin=torch.min(data),vmax=torch.max(data))

    # colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if(only_imshow==False):
        divider = make_axes_locatable(main_ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(im, ax=main_ax,cax=cax)
        cb.set_label(label='Transmission rate \nper voxel',fontsize=14)

        # Axis labels 
        main_ax.set_aspect('equal')
        main_ax.set_xlabel('y [mm]',fontsize=14)
        main_ax.set_ylabel('x [mm]',fontsize=14)
        main_ax.set_aspect("equal")

        # ticks Y
        set_ticks_and_label(n_ticks=4,
                            voi_centers=VOI.voxel_centers[:,0,0,0],
                            n_vox=VOI.n_vox_xyz[0],ax=main_ax,dim=1,fontsize=fontsize)
        # ticks X
        set_ticks_and_label(n_ticks=8,
                            voi_centers=VOI.voxel_centers[0,:,0,1],
                            n_vox=VOI.n_vox_xyz[1],ax=main_ax,dim=0,fontsize=fontsize)

    else:
        main_ax.axis('off')
        main_ax.text(0, 0, 'z = {} mm'.format(VOI.voxel_centers[0,0,z_slice,2]), bbox={'facecolor': 'white', 'pad': 10})
        
        
def plot_transmission_score_slices(transmission:Tensor, VOI:VolumeInterest) -> None:
    
    r"""
    Plot the transmission score of a voxelized VOI.
    
    INPUT:
     - VOI:VolumeInterest The voxelized volume of interest used for tracking and reconstruction.
     - transmission: Tensor with size (n_vox_x,n_vox_y,n_vox_z) containing the scores of the reconstruction.
     
    NOTE: this function requires the imshow_transmission_rate method.
    """
    
    fig,ax = plt.subplots(nrows=int(len(transmission[0,0,:])/3),ncols=3,figsize=(16,20))
    ax = ax.ravel()
    for i in range(len(transmission[0,0,:])):
        imshow_transmission_rate(1/transmission[:,:],fig=fig,main_ax=ax[i],VOI=VOI,fontsize=0,only_imshow=True,z_slice=i)    


def plot_lumuosity_summary(transmission:Tensor,free_sky:Tensor,VOI:VolumeInterest)->None:
    
    r"""
    INPUTS:
    
     - transmission: Tensor with size(n_vox_x,n_vox_y,n_vox_z) containing the backprojection scores 
    for target measurement.
    
     - free_sky: Tensor with size(n_vox_x,n_vox_y,n_vox_z) contaning the backprojection scores 
    for free sky measurement.

     - VOI:VolumeInterest, the volume of interest used for the tracking and reconstruction.
    
    """
    # Density distribution of muon hit per voxels
    maximum = torch.max(torch.cat((transmission,free_sky)).ravel()).item()
    mean = int(transmission.ravel().mean().item())

    fig,ax = plt.subplots(ncols=2,figsize=(15,5))
    fig.suptitle("Density distribution of muon hit per voxel",alpha=.8)
    ax[0].set_title("Target measurement")
    ax[0].hist(transmission.ravel(),bins=80,density=True,alpha=.7,range=(0,maximum))
    ax[0].axvline(x=mean,label="mean = {}".format(mean),color='red')
    ax[0].set_xlabel("# muon hit per voxel")
    ax[0].set_ylabel("Normalized density")
    ax[0].legend()

    mean = int(free_sky.ravel().mean().item())

    ax[1].set_title("Free-sky measurement")
    ax[1].hist(free_sky.ravel(),bins=80,density=True,alpha=.7,range=(0,maximum),color='green')
    ax[1].axvline(x=mean,label="mean = {}".format(mean),color='red')
    ax[1].set_xlabel("# muon hit per voxel")
    ax[1].set_ylabel("Normalized density")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    # Total number of hit per volume
    vox_volume_cm = (VOI.vox_width/10)**3
    vox_volume_m = (VOI.vox_width/1000)**3

    n_vox = np.prod(VOI.n_vox_xyz)
    n_hit_free_sky, n_hit_target = free_sky.ravel().sum().int(), transmission.ravel().sum().int()

    print("VOI volume = {} cm3, {:.2f} m3".format(n_vox*vox_volume_cm,n_vox*vox_volume_m))
    print("Total number of hit voxels = {} for free sky, {} for target".format(n_hit_free_sky,n_hit_target))
    print("Number of hit per cm3 = {:.2f}".format(n_hit_target/(n_vox*vox_volume_cm)))
    print("Number of hit per voxel (target measurement)= {:.2f}".format(n_hit_target/n_vox))
    print("Number of hit per voxel (free sky measurement)= {:.2f}".format(n_hit_free_sky/n_vox))


