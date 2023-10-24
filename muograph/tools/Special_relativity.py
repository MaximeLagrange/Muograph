import numpy as np
import math
import torch
from torch import Tensor

muon_mass = 105.7 #MeV

def momentum_from_kinetic_E(kinetic,mass=105.7):

    return torch.sqrt(kinetic**2+2*mass*kinetic)

# def kinetic_E_from_momentum(momentum,mass = 105.7):
#     return 

class particle:

    def __init__(self,mass,kinetic_E):

        self.mass = mass
        self.kinetic_E = kinetic_E
        self.momentum = self.momentum_from_kinetic_E()

    def momentum_from_kinetic_E(self):
        '''
        Computes particle momentum in MeV/c from its kinetic energy in MeV 
        '''
        return math.sqrt(self.kinetic_E**2 + 2*self.kinetic_E*self.mass)


    

    