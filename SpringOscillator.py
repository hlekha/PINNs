#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy import pi
from scipy import constants
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
# import random

"""
Here we will generate a dataset to retrieve observational data
which our PINN will receive as an input
"""

def actual(t, m, k, A):
    inside = math.sqrt(k/m)*t
    return A*torch.cos(inside)


# we only use 13 points because we want PINN to be able to solve from scarce data
t_data = torch.linspace(0, 10, 13).unsqueeze(1) 
t_physics = torch.linspace(0, 10, 100, requires_grad=True).unsqueeze(1)

x_data = actual(t_data, 1.0, 4.0, 1.0)

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


        #self.k = nn.Parameter(
        #    torch.tensor(float(random.randint(0, 100)))
        #)

        self.k = nn.Parameter(torch.tensor(2.0))

    def forward(self, t):
        x = self.model(t)

        return x


"""
Checking if the predicted motion obeys Newton's 
Second Law and Hooke's Law
"""

def data_loss(model, times, positions):
    x = model(times)

    return torch.mean((x - positions)**2)

def physics_loss(model, times, m, scale=1.0):
    x = model(times)

    v = torch.autograd.grad(
        x, times,
        grad_outputs=torch.ones_like(x),
        create_graph=True # allows second differentiation by tracking operations
    )[0]

    a = torch.autograd.grad(
        v, times,
        grad_outputs=torch.ones_like(v),
        create_graph=True
    )[0]

    r_t = m*a + model.k*x
    loss_physics = scale * torch.mean(r_t)

    return loss_physics

def _get_total_loss(model, t_data, x_data, t_physics, m, scale):
    loss_data = data_loss(model, t_data, x_data)
    loss_physics = physics_loss(model, t_physics, m, scale)

    total_loss = loss_data + loss_physics
    return total_loss

model = SpringPINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5000
t_plot = torch.linspace(0, 10, 300).unsqueeze(1)

prediction_history = []
k_history = []
epoch_history = []
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()

    loss = _get_total_loss(model, t_data, x_data, t_physics, 1.0, 1.0)

    loss.backward()
    optimizer.step()

    if epoch % 50 == 0 or epoch == epochs - 1:
        with torch.no_grad():
            prediction = model(t_plot).detach().cpu().numpy().flatten()

        prediction_history.append(prediction)
        k_history.append(model.k.item())
        epoch_history.append(epoch)
        loss_history.append(loss.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, k = {model.k.item():.4f}")



# In[ ]:




