#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy import pi
from scipy import constants
"""
Here we will generate a dataset to retrieve observational data
which our PINN will receive as an input
"""

def actual(t, m, k, A):
    inside = torch.sqrt(k/m)*t
    return A*torch.cos(inside)
time = torch.linspace(0, 10, 13) # we only use 13 points because we want PINN to be able to solve from scarce data


class SpringPINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )


        self.k = nn.Parameter(
            torch.tensor(
                float(random.randint(0, 100))
            )
        )

    def forward(self, t):
        x = self.model(t)

        return x


"""
Checking if the predicted motion obeys Newton's 
Second Law and Hooke's Law
""""

def data_loss(self
def physics_loss(x, m, scale):
    v = torch.autograd.grad(
        x, t,
        grad_outputs=torch.ones_like(x),
        create_graph=True # allows second differentiation by tracking operations
    )[0]

    a = torch.autograd.grad(
        v, t,
        grad_outputs=torch.ones_like(v),
        create_graph=True
    )[0]

    r_t = m*a + k*x
    loss_physics = scale * r_t

    return r_t


# In[ ]:




