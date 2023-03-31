# -*- coding: utf-8 -*-
"""
"""
import torch
import scipy.integrate
import numpy as np
import os, sys
import matlab.engine

if 'windows' in sys.platform:
    bar = '\\'
else:
    bar = '/'

def matlab_interface():                                                         # creates Matlab-Python interface
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir) 
    eng = matlab.engine.start_matlab()
    eng.addpath(parent_dir + bar + "SimulationFramework", nargout= 0)
    eng.addpath(parent_dir + bar + "SimulationFramework" + bar + "background", 
                nargout= 0)
    eng.addpath(parent_dir + bar + "SimulationFramework" + bar + "background" 
                + bar + "utils", nargout= 0)
    eng.addpath(parent_dir + bar + "SimulationFramework" + bar + "background"
                + bar + "CrossSectionData", nargout= 0)
    eng.addpath(parent_dir + bar + "SimulationFramework" + bar + "test_results",
                nargout= 0)
    return eng


class ObjectView(object):                                                       # simplifies accessing the hyperparameters
    def __init__(self, d): 
        self.__dict__ = d
    

def integrate_model(model, t_span, y0, use_torch=True, **kwargs):        # integrates the generated NN models over time
    def fun(t, x):
        if use_torch:
            x = torch.tensor(x, requires_grad=True, 
                             dtype=torch.float32).reshape(1,len(y0))
            t = torch.zeros_like(x[...,:1])
        else:
            x = x.reshape(1,len(y0))
            t = np.zeros_like(x[...,:1])
        dx = model(x, t=t).reshape(-1)
        if use_torch:
            dx = dx.data.numpy()
        return dx
    return scipy.integrate.solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)


def normalize(data_raw, x_denorm, norm_range):                                  # normalizes the features
    x_norm = np.empty(x_denorm.shape)
    n_DoF = int(x_norm.shape[1]/2)
    for i in range(n_DoF):
        # x_norm[:,i] = (x_denorm[:,i] - np.min(data_raw[:,:]))/np.ptp(data_raw)  
        x_norm[:,i] = np.ptp(norm_range)*(x_denorm[:,i] - np.min(data_raw[:,:n_DoF]))/np.ptp(data_raw[:,:n_DoF]) + norm_range[0]   
        x_norm[:,n_DoF+i] = np.ptp(norm_range)*(x_denorm[:,n_DoF+i] - np.min(data_raw[:,n_DoF:]))/np.ptp(data_raw[:,n_DoF:]) + norm_range[0]   
    return x_norm


def denormalize(data_raw, x_norm, norm_range):                                  # denormalizes the features
    x_denorm = np.empty(x_norm.shape)
    n_DoF = int(x_denorm.shape[1]/2)
    for i in range(n_DoF):
        # x_denorm[:,i] = x_norm[:,i]*np.ptp(data_raw) + np.min(data_raw)
        x_denorm[:,i] = (x_norm[:,i] - norm_range[0])*np.ptp(data_raw[:,:n_DoF])/np.ptp(norm_range) + np.min(data_raw[:,:n_DoF])
        x_denorm[:,n_DoF+i] = (x_norm[:,n_DoF+i] - norm_range[0])*np.ptp(data_raw[:,n_DoF:])/np.ptp(norm_range) + np.min(data_raw[:,n_DoF:])
    return x_denorm


def standardize(data_raw, x_destand):                                           # standardize the features
    x_stand = np.empty(x_destand.shape)
    for i in range(x_destand.shape[1]):
        x_stand[:,i] = (x_destand[:,i] - np.mean(data_raw[:,i]))/np.std(data_raw[:,i]) 
    return x_stand


def destandardize(data_raw, x_stand):                                           # destandardizes the features
    x_destand = np.empty(x_stand.shape)
    for i in range(x_destand.shape[1]):
        x_destand[:,i] = x_stand[:,i]*np.std(data_raw[:,i]) + np.mean(data_raw[:,i]) 
    return x_stand
    

def post_process(X_sol, data_raw, model_param, *args):
    for arg in args:
        phi_r = arg
    if model_param.normalize:                                                   # denormalizes/destandardizes solution (if enabled):
        X = denormalize(data_raw['x'], X_sol, model_param.norm_range)
        if model_param.model == 'FOM':
            p, q = np.split(X, 2, axis=1)  
        elif model_param.model == 'ROM':   
            etap, eta = np.split(X, 2, axis=1)
            p, q = (phi_r@etap.T).T, (phi_r@eta.T).T
    elif model_param.standardize:                                                       
        X = destandardize(data_raw['x'], X_sol)
        if model_param.model == 'FOM':
            p, q = np.split(X, 2, axis=1)  
        elif model_param.model == 'ROM':   
            etap, eta = np.split(X, 2, axis=1)
            p, q = (phi_r@etap.T).T, (phi_r@eta.T).T        
    else:                                                      
        X = X_sol
        if model_param.model == 'FOM':
            p, q = np.split(X, 2, axis=1)  
        elif model_param.model == 'ROM':   
            etap, eta = np.split(X, 2, axis=1)
            p, q = (phi_r@etap.T).T, (phi_r@eta.T).T
    return p, q
        
            
            




    