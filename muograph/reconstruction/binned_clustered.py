# Usual suspects
from typing import Dict, List, Union, Tuple, Optional
import torch
import functools
from functools import partial
import math
NoneType = type(None)
from fastprogress import progress_bar
from copy import deepcopy

# Muograph imports
from tracking.Tracking import *
from volume.Volume_Interest import *
from reconstruction.POCA import * 

def normalize(s:torch.tensor) -> torch.tensor:
    
    s_max, s_min = torch.max(s), torch.min(s)
    
    return (s-s_min)/(s_max-s_min)

class BCA(POCA):
    
    def __init__(self, tracks:Tracking, output_dir:str, cut_low_theta:float=0.0001) -> None:
        
        """
        Set up the Binned clustered Algorithm.
        
        INPUT:
         - tracks:Tracking, an instance of the Tracking class.
         - output_dir:str, path to the directory where to save figures abd results.
         - cut_low_theta:0.001
        """
        # POCA initialization
        super().__init__(tracks,output_dir,cut_low_theta)
        
        # Copy tracks
        self.bca_tracks = deepcopy(self.tracks)
        # Copy POCA points
        self.bca_POCA_points = deepcopy(self.POCA_points)
        # Copy POCA points voxels indices
        self.bca_indices = deepcopy(self.indices)
        
        # Directory
        self.output_dir = output_dir
        self.create_directory(self.output_dir)
        self.save(self.output_dir)

     
    def compute_distance_2_points(self, pocas:torch.tensor) -> torch.Tensor:
        
        """Copmute the distance between each of the given poca points.
        
        INPUT:
         - pocas: torch.tensor with size (N,3)
         
        OUTPUT:
         - distances:torch.tensor with size (N,N), a symmetric matrix with diagonal 0. 
         Contains distances between each poca points.
        """
        
        pocas = pocas.reshape((pocas.size()[0],3,1))
        return torch.sqrt(torch.sum(torch.square(pocas - pocas.T), axis=1))
    
        
    def compute_scattering_momentum_weight(self, 
                                           dtheta:torch.Tensor,
                                           p:Union[torch.Tensor,NoneType] = None) -> torch.Tensor:

        """Compute weights based on muon scattering angle and momentum (if availabe).

        INPUT:
         - dtheta: torch.tensor with size (N,3), muon scattering angle.
         
         - p: torch.tensor with size (N,3), muon momentum if available.

        OUTPUT:
        
         - weights:torch.tensor with size (N,N), a symmetric matrix. 
         If momentum is available, computed as p*dtheta. If not, computed as dtheta.
        """

        if p is not None:
            dtheta_p = dtheta*p
            return dtheta_p * dtheta_p.T
        else:
            return dtheta * dtheta.T
        
        
    def compute_low_theta_events_voxel_wise_mask(self,n_max_per_voxel:int) -> Tuple[torch.tensor]:
        
        """
        Only keep the n-th highest scattering event among poca points within a given voxel. Other events will be removed.
        
        INPUT:
        
         - n_max_per_voxel:int, the number of highest scattering angle to keep among poca points located within a given voxel.
        
        OUTOUT:
        
         - mask:torch.tensor with size (N_rejected_event), a mask rejecting events.
         
         - nhit:torch.tensor with size (Nx,Ny,Nz,1) with Ni the number of voxels along a given dimension. 
         nhits[i,j,k] is the number of poca points within voxel i,j,k, BEFORE removing events.
         
         - nhit_cut:torch.tensor with size (Nx,Ny,Nz,1) with Ni the number of voxels along a given dimension. 
         nhits[i,j,k] is the number of poca points within voxel i,j,k, AFTER removing events.
        """
        
        rejected_events = []
        nhit = torch.zeros([self.VOI.n_vox_xyz[0],self.VOI.n_vox_xyz[1],self.VOI.n_vox_xyz[2],1],dtype=torch.int32)
        nhit_cut = torch.zeros([self.VOI.n_vox_xyz[0],self.VOI.n_vox_xyz[1],self.VOI.n_vox_xyz[2],1],dtype=torch.int32)
        
        for i in progress_bar(range(self.VOI.n_vox_xyz[0])):
            for j in range(self.VOI.n_vox_xyz[1]):
                for k in range(self.VOI.n_vox_xyz[2]):
                    
                    # Sort muons as a function of their scattering angle
                    mask = ((self.bca_indices==torch.tensor([i,j,k])).sum(dim=-1)==3)
                    sorted_dtheta, order = torch.sort(self.bca_tracks.dtheta[mask],descending=True)
                    muon_id = mask.nonzero()
                    
                    # Compute number of POCA point per voxel before cut
                    nhit[i,j,k] = len(sorted_dtheta)
                    
                    # Only keep events with the n highest scattering angle
                    if len(sorted_dtheta)>n_max_per_voxel:
                        rejected_events+=muon_id[order[n_max_per_voxel:]].tolist()
                        nhit_cut[i,j,k] = n_max_per_voxel
        
        rejected_events, _ = torch.sort(torch.tensor(rejected_events).reshape(len(rejected_events)))
        events = torch.tensor(range(self.bca_tracks.n_event_masked))
        mask = torch.isin(events, rejected_events)==False
        return mask, nhit, nhit_cut

        
    def compute_vox_wise_metric(self,
                                metric_method:Union[functools.partial,NoneType],
                                vox_id:torch.Tensor,
                                use_p:bool) -> None:
        
        """
        Computes a voxel-wise scattering density metric.
        CREDITS: A binned clustering algorithm to detect high-Z material using cosmic muons, 
                 2013 JINST 8 P10013, (http://iopscience.iop.org/1748-0221/8/10/P10013)
                 
        INPUT:
         - vox_id:torch.Tensor(dtypre=torch.int32), the voxel indices.
         
        OUTPUT:
         - full_metric:torch.Tensor with size N, where N is the number of poca points 
         within the voxel with indices vox_id and satisfies the condition on p and dtheta. 
         Zero elements are removed.
        
        """
        
        # Mask events outside the voxel
        poca_in_vox_mask = (self.bca_indices==vox_id).sum(dim=-1)==3
        
        # POCA points within the voxel
        poca_in_vox = self.bca_POCA_points[poca_in_vox_mask]
        
        # Distance metric
        distance_metric = self.compute_distance_2_points(pocas=poca_in_vox)
        
        # Scattering metric
        if(use_p):
            scattering_weights = self.compute_scattering_momentum_weight(dtheta = self.bca_tracks.dtheta[poca_in_vox_mask],
                                                                         p = self.bca_tracks.p[poca_in_vox_mask])
        else:
            scattering_weights = self.compute_scattering_momentum_weight(dtheta = self.bca_tracks.dtheta[poca_in_vox_mask])
        # Keep only lower triangle element within symmetric matrix
        full_metric = torch.tril(torch.where((distance_metric!=0)&(scattering_weights!=0),distance_metric/scattering_weights,0.))
        if metric_method is not None:
            return metric_method(full_metric[full_metric!=0.])
        else:
            return full_metric[full_metric!=0.]
            
    
    def compute_voxels_distribution(self,
                                    metric_method:Union[functools.partial,None],
                                    use_p:bool, 
                                    n_min_per_vox:int=2) -> List:
        
        """
        Compute voxel-wise weight distribution, according to the self.compute_vox_wise_metric() method.
        
        INPUT:
         
         - use_p:bool, if true, momentum will be used in the voxel weight computation, 
         handled by self.compute_vox_wise_metric(). If False, only scattering angle will be used.
        
         - n_min_per_voxel:int, the min number of POCA point per voxel. 
         If a voxel contains less than n_min_per_voxel, is final score will be 0.
         
        OUTPUT:
         
         - score_list:List the list containing a list of scores for each voxel.

        """
        
        from fastprogress import progress_bar

        # Create a list with size [Nvox_x,Nvox_y,Nvox_z]
        score_list = torch.zeros((self.VOI.n_vox_xyz[0],self.VOI.n_vox_xyz[1],self.VOI.n_vox_xyz[2])).tolist()

        # Assign an empty list [] to each element of score_list
        print('\nAssigning score to voxels:')
        for i in progress_bar(range(self.VOI.n_vox_xyz[0])):
            for j in range(self.VOI.n_vox_xyz[1]):
                for k in range(self.VOI.n_vox_xyz[2]):
                    
                    # compute distribution if there is at least n poca point inside voxel
                    if (((self.bca_indices == torch.tensor([i,j,k],dtype=torch.int32)).sum(dim=-1)==3).sum()>n_min_per_vox):
                        score_list[i][j][k] = self.compute_vox_wise_metric(metric_method=metric_method,
                                                                           vox_id=torch.tensor([i,j,k]),
                                                                           use_p = use_p)
                    else:
                        score_list[i][j][k] = []
                    
        return score_list
    
    
    def compute_final_scores(self, score_list:List, score_method:functools.partial) -> Tuple[torch.Tensor]:
        
        """
        Compute voxel-wise scores from a voxel-wise list of scores.
        
        INPUT:
         - score_list:List the list containing a list of scores for each voxel.
         - score_method:functools.partail the function use to convert the score list into a float.
         
        OUTPUT:
         - final_voxel_scores:torch.Tensor containing the final voxel score with size (Nx,Ny,Nz).
         Where Ni the number of voxels along a certain axis.
         
         - hit_per_voxel:torch.Tensor conatining the number of POCA points within each voxel, 
         with size (Nx,Ny,Nz).
        """
        
        
        Nx, Ny, Nz = self.VOI.n_vox_xyz[0], self.VOI.n_vox_xyz[1], self.VOI.n_vox_xyz[2]
        final_voxel_scores, hit_per_voxel = torch.zeros((Nx,Ny,Nz)), torch.zeros((Nx,Ny,Nz)) 
        
        print('Computing final voxels score:')
        for i in progress_bar(range(Nx)):
            for j in range(Ny):
                for k in range(Nz):
                    
                    if(len(score_list[i][j][k])>0):
                        hit_per_voxel[i,j,k] += len(score_list[i][j][k])
                        final_voxel_scores[i,j,k] = score_method(score_list[i][j][k])
                    else:
                        score_list[i][j][k] = 0.
                        
        return final_voxel_scores, hit_per_voxel
    

    def cut_events(self,mask:torch.tensor) -> None:
        
        """
        Only keep muons events for which mask == True.
        """
        
        self.bca_indices = self.bca_indices[mask]
        self.bca_tracks.Apply_mask(mask)
        self.bca_POCA_points = self.bca_POCA_points[mask]
    
    
    def bca_reconstruction(self,
                           n_max_per_vox:int,
                           score_method:functools.partial = partial(np.quantile,q=.5),
                           metric_method:Union[functools.partial,NoneType] = partial(np.log),
                           p_range:Tuple[int] = [0,250000],
                           dtheta_range:Tuple[float] = [0,math.pi/3],
                           use_p:bool = False, 
                           n_min_per_vox:int=2,
                           save:bool = False,
                           plot:bool = True) -> None:
        
        """
        Run the BCA algorithm, as implemented in: 
        A binned clustering algorithm to detect high-Z material using cosmic muons, 
        2013 JINST 8 P10013, 
        (http://iopscience.iop.org/1748-0221/8/10/P10013).
        
        INPUT:
        
         - n_max_per_voxel:int, the number of highest scattering angle to keep among poca points located within a given voxel.
        
         - score_method:functools.partail the function use to convert the score list into a float.
         
         - p_range:Tuple[int]=[0, 1,000,000], the momentum range in MeV/c. 
         Muon weights with p outside of p_range are set to 0.
         
         - dtheta_range:Tuple[int]=[0, math.pi/2], the scattering angle range in radiants. 
         Muon weights with dtheta outside of dtheta_range are set to 0.
         
         - use_p:bool if False, momentum is not used during metric computation.
         
         - n_min_per_voxel:int, the min number of POCA point per voxel. 
         If a voxel contains less than n_min_per_voxel, is final score will be 0.
         
         OUTPUT:
         
          - final_voxel_scores:torch.Tensor containing the final voxel score with size (Nx,Ny,Nz).
         Where Ni the number of voxels along a certain axis.
         
         - hit_per_voxel:torch.Tensor conatining the number of POCA points within each voxel, 
         with size (Nx,Ny,Nz).
        """
        
        def get_partial_name_args(func:Union[functools.partial,NoneType]) -> Union[str]:
            
            if(func is not None):
            
                func_name = func.func.__name__
                args, values = list(func.keywords.keys()), list(func.keywords.values())

                for i, arg in enumerate(args):

                    func_name+="_{}={}".format(arg,values[i])

                return func_name
            else:
                return "None"
        
        # store BCA parameters
        self.bca_params = {"n_max_per_vox":n_max_per_vox, 
                           "score_method":get_partial_name_args(score_method),
                           "metric_method":get_partial_name_args(metric_method),
                          "p_range":p_range,
                          "dtheta_range":dtheta_range,
                          "use_p":use_p,
                          "n_min_per_vox":n_min_per_vox}
        
        # get BCA name from its parameters
        self.bca_name = self.get_bca_name()
        
        # create output directory with BCA name
        self.dir_name = self.output_dir + self.bca_name + "/"
        self.create_directory(dir_name=self.dir_name)
        
        # copy relevant features before event selection
        self.bca_tracks = deepcopy(self.tracks)
        self.bca_indices = deepcopy(self.indices)
        self.bca_POCA_points = deepcopy(self.POCA_points)
        
        # keep only the n poca points with highest scattering angle within a voxel
        self.mask, self.nhit, self.nhit_rejected = self.compute_low_theta_events_voxel_wise_mask(n_max_per_voxel=n_max_per_vox)
        self.cut_events(self.mask)
        
        # momentum cut
        if (use_p):
            p_mask = (self.bca_tracks.p>p_range[0])&(self.bca_tracks.p<p_range[1])
        else:
            p_mask = torch.ones_like(self.bca_tracks.dtheta,dtype=torch.bool)
        
        # scattering angle cut
        dtheta_mask = (self.bca_tracks.dtheta>dtheta_range[0])&(self.bca_tracks.dtheta<dtheta_range[1])
        
        # apply dtheta, p cuts
        self.cut_events(mask=p_mask&dtheta_mask)
        
        # compute voxels distribution
        self.score_list = self.compute_voxels_distribution(metric_method=metric_method,
                                                          use_p=use_p, 
                                                          n_min_per_vox=n_min_per_vox)
        
        # compute fina scores
        self.final_scores, self.hit_per_voxel = self.compute_final_scores(score_list=self.score_list, score_method=score_method)
        
        # normalize scores
        self.normalized_scores = normalize(self.final_scores)
        
        # replace 0 values by max(final_scores)
        self.normalized_scores = torch.where(self.normalized_scores==0,
                                             torch.max(self.normalized_scores),
                                             self.normalized_scores)
        # plot BCA summary
        self.plot_summary()
        
        if save:
            self.save(directory=self.dir_name)
            
        if plot:
            from plotting.Plotting import plot_VOI_pred
            plot_VOI_pred(self.normalized_scores,reverse=True,filename=self.dir_name+"bca_plot")
            
    
    def plot_param_summary(self, ax, save:bool) -> None:
        
        plt.axis("off")
        ax.set_title("BCA parameter summary")
        values = list(self.bca_params.values())
        for i,key in enumerate(self.bca_params.keys()):
            ax.text(0,.8-(i/10),"{} :".format(key))
            if type(values[i])==float:
                ax.text(.4,.8-(i/10),"{:.2f}".format(values[i]))
            else:
                ax.text(.4,.8-(i/10),"{}".format(values[i]))
        if save:
            plt.savefig(self.dir_name+"parameter_summary")
        plt.show()
       
    
    def plot_summary(self,save:bool=True) -> None:
        
        # Plot BCA parameters 
        fig,ax = plt.subplots()
        self.plot_param_summary(ax=ax,save=save)
        
        # Plot scattering angle and momentum summary
        fig,ax = plt.subplots(ncols=2,nrows=2,figsize=(11,7))
        ax = ax.ravel()

        # Scattering angle
        ax[0].set_title("Scattering angle distribution")
        ax[0].hist(self.bca_tracks.dtheta,
                bins=100,
                log=True,
                density=True,
                range=(0,math.pi/2),
                alpha=.4,
                label="selection: {}".format(len(self.bca_tracks.dtheta)))

        ax[0].hist(self.tracks.dtheta,
                bins=100,
                log=True,
                density=True,
                range=(0,math.pi/2),
                alpha=.4,
                label="all: {}".format(len(self.tracks.dtheta)))

        ax[0].axvline(x=self.bca_params["dtheta_range"][0],
                   label=r"$\theta_{low} = "+"{:.3f}$".format(self.bca_params["dtheta_range"][0]),
                   color="green")

        ax[0].axvline(x=self.bca_params["dtheta_range"][1],
                   label=r"$\theta_{high} = "+"{:.3f}$".format(self.bca_params["dtheta_range"][1]),
                   color="red")
        ax[0].set_xlabel(r"Scattering angle $\Delta\theta$ [rad]")
        ax[0].legend()

        # Momentum
        ax[1].set_title("Momentum distribution")
        ax[1].hist(self.bca_tracks.p,
                bins=100,
                log=True,
                density=True,
                range=(0,self.tracks.p.max().numpy()),
                alpha=.4,
                label="selection: {}".format(len(self.bca_tracks.p)))

        ax[1].hist(self.tracks.p,
                bins=100,
                log=True,
                density=True,
                range=(0,self.tracks.p.max().numpy()),
                alpha=.4,
                label="all: {}".format(len(self.tracks.p)))

        ax[1].axvline(x=self.bca_params["p_range"][0],
                   label=r"$p_{low} = "+"{:.3f}$".format(self.bca_params["p_range"][0]),
                   color="green")

        ax[1].axvline(x=self.bca_params["p_range"][1],
                   label=r"$p_{high} = "+"{:.3f}$".format(self.bca_params["p_range"][1]),
                   color="red")
        ax[1].set_xlabel(r"Momentum $[MeV/c]$")
        ax[1].legend()

        # Low sattering angle
        ax[2].set_title("Scattering angle distribution (low)")
        ax[2].hist(self.bca_tracks.dtheta,
                bins=100,
                log=True,
                density=True,
                range=(0,5*math.pi/180),
                alpha=.4,
                label="selection: {}".format(len(self.bca_tracks.dtheta)))

        ax[2].hist(self.tracks.dtheta,
                bins=100,
                log=True,
                density=True,
                range=(0,5*math.pi/180),
                alpha=.4,
                label="all: {}".format(len(self.tracks.dtheta)))

        ax[2].axvline(x=self.bca_params["dtheta_range"][0],
                   label=r"$\theta_{low} = "+"{:.3f}$".format(self.bca_params["dtheta_range"][0]),
                   color="green")

        ax[2].set_xlabel(r"Scattering angle $\Delta\theta$ [rad]")
        ax[2].legend()

        # Low momentum
        ax[3].set_title("Momentum distribution (low)")
        ax[3].hist(self.bca_tracks.p,
                bins=100,
                log=True,
                density=True,
                range=(0,10000),
                alpha=.4,
                label="selection: {}".format(len(self.bca_tracks.p)))

        ax[3].hist(self.tracks.p,
                bins=100,
                log=True,
                density=True,
                range=(0,10000),
                alpha=.4,
                label="all: {}".format(len(self.tracks.p)))

        ax[3].axvline(x=self.bca_params["p_range"][0],
                   label=r"$p_{low} = "+"{:.3f}$".format(self.bca_params["p_range"][0]),
                   color="green")

        ax[3].set_xlabel(r"Momentum $[MeV/c]$")
        ax[3].legend()

        plt.tight_layout()
        
        # Save figure
        if (save):
            plt.savefig(self.dir_name+"scattering_angle_momentum_summary")

        plt.show()
        
        
    def plot_voxel_distribution(self,indices:Tuple[int],save:bool=True) -> None:
        
        i,j,k = indices[0],indices[1],indices[2]
        distribution = self.score_list[i][j][k]
        fig,ax = plt.subplots()
        fig.suptitle("voxel {},{},{} distribution".format(i,j,k))
        ax.hist(distribution,bins=50,color="brown")
        plt.axvline(x = np.median(distribution),color="blue",label=("median = {:.2f}".format(np.median(distribution))))
        plt.legend()
        plt.savefig(self.dir_name+"voxel_{}_{}_{}_distribution".format(i,j,k))
        plt.show()
        
        
    def get_bca_name(self,) -> str:
        
        method = "method_{}_".format(self.bca_params["score_method"])
        metric = "metric_{}_".format(self.bca_params["metric_method"])
        dtheta = "{:.2f}_{:.2f}_rad_".format(self.bca_params["dtheta_range"][0],self.bca_params["dtheta_range"][1])
        dp = "{:.0f}_{:.0f}_MeV_".format(self.bca_params["p_range"][0],self.bca_params["p_range"][1])
        n_min_max = "n_min_max_{}_{}_".format(self.bca_params["n_min_per_vox"],self.bca_params["n_max_per_vox"])
        use_p = "use_p_{}".format(self.bca_params["use_p"])
        
        bca_name = method+metric+dtheta+dp+n_min_max+use_p
        
        return bca_name
    
   
    def create_directory(self, dir_name:str) -> None:
        
        import os
        if(os.path.exists(dir_name)==False):
            os.mkdir(dir_name)
            print("\n{} directory created".format(dir_name))
        
        
    def save(self,directory:str) -> None:
        
        import pickle
        
        with open(directory+"bca_class", "wb") as f:
            pickle.dump(self,f)
        
