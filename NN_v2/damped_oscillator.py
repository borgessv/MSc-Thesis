#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 01:25:52 2023

@author: vitor
"""

import numpy as np
from scipy.integrate import solve_ivp

def eom(t, y, m, k, c):
    """
    Define the equations of motion for a damped mass-spring system
    """
    p, q = y
    dqdt = p/m
    dpdt = -k*q - c*dqdt
    return [dpdt, dqdt]

def dataset(m=1, k=1, c=0.1, num_samples=1000, t_span=(0, 5), dt=0.1, noise_std=0.01):
    """
    Create a dataset of generalized coordinates, momenta, and their derivatives for a damped mass-spring system
    """
    t_eval = np.linspace(t_span[0], t_span[1],int((t_span[1]-t_span[0])//dt)+1)
    dataset = {'x': [], 'dx': []}
    for _ in range(num_samples):
        # Generate random initial conditions
        q0 = np.random.uniform(-1.0, 1.0)
        p0 = np.random.uniform(-1.0, 1.0)
        y0 = [p0, q0]
        
        # Solve the equations of motion numerically
        sol = solve_ivp(lambda t, y: eom(t, y, m, k, c), t_span, y0, t_eval=t_eval)
        # Store the generalized coordinates and their derivatives
        p = sol.y[0].T
        q = sol.y[1].T
        dpdt = -k*q - c*p
        dqdt = p/m
        
        q += np.random.randn(*q.shape)*noise_std
        p += np.random.randn(*p.shape)*noise_std
        
        # Append to the dataset
        dataset['x'].append(np.column_stack((p, q)))
        dataset['dx'].append(np.column_stack((dpdt, dqdt)))
    
    # Convert to NumPy arrays
    dataset['x'] = np.array(dataset['x'])
    dataset['dx'] = np.array(dataset['dx'])
    dataset = {'x': np.concatenate(dataset['x']), 
                'dx': np.concatenate(dataset['dx']),
                'phi_r': []}   
    return dataset

def hamiltonian(p, q, m, k, c):
    H = []
    for i in range(len(p)):
        dqdt = eom(0, [p[i], q[i]], m, k, c)[1]
        H.append(0.5*(p[i]**2/m + k*q[i]**2 + c*dqdt**2))
    return H

    

# # Define the parameters
# m = 1.0
# k = 2.0
# c = 0.0
# num_samples = 1000

# # Create the dataset
# data_raw = damped_oscillator_dataset(m, k, c, num_samples)

# sio.savemat('damped_oscillator.mat', data_raw)
