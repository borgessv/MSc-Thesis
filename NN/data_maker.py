# -*- coding: utf-8 -*-
"""
Description:
Dataset creation using simulation outputs of the dynamical system modeled by  
the lumped-parameter method combined with the Hamiltonian mechanics framework.
For more details check MATLAB's 'simulation.m' code.

Author: Vitor Borges Santos - borgessv93@gmail.com
Version: 10-July-2022
"""

import os, sys
import pickle
import scipy.io as sio
import numpy as np
import pandas as pd
import matlab.engine

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

if 'windows' in sys.platform:
    bar = '\\'
else:
    bar = '/'

def make_dataset(model_param, seed=None, samples=20, **kwargs):
    print('\rCreating dataset... {:.2f}% done'.format(0), end='')
     
    eng = model_param.matlab_engine                                             # defines model parameters 
    model = model_param.model
    n_modes = model_param.n_modes                                               # only if ROM
    DoF = model_param.DoF
    gravity = model_param.gravity
    disp_progress = model_param.disp_progress
    samples = model_param.n_samples 
    
    beam_data = model_param.beam_data                                           # initializes structure properties
    beam = pd.read_excel(parent_dir+bar+"SimulationFramework"+bar+beam_data)
    n_DoF = len(DoF)*int(np.sum(np.array(beam.iloc[3,1:])))
    M, I, K, C = eng.structure_properties(beam_data,DoF,disp_progress,nargout=4)
    if model == "ROM":
        x0 = matlab.double(np.zeros((n_DoF,1)))
        Xeq = eng.solve_equilibrium(K,DoF,gravity,model,x0,disp_progress,nargout=1)
        phi_r = eng.create_rom(n_modes,DoF,Xeq,disp_progress,nargout=1)
    else:
        phi_r = []
    
    dt = model_param.timestep                                                   # simulation time definitions
    t0 = model_param.tspan[0]
    t1 = model_param.tspan[1]
    tvec = np.array([np.linspace(t0,t1,int((t1-t0)/dt)+1)])
    teval = matlab.double(tvec)

    np.random.seed(seed)                                                        # initial condition parameters
    p0span = model_param.p0span
    q0span = model_param.q0span
    
    X_data, Xdot_data, t_data = [], [], []                                      # dataset creation loop
    for i in range(samples):
        p0 = (p0span[1]-p0span[0])*np.random.rand(n_DoF) + p0span[0] 
        q0 = (q0span[1]-q0span[0])*np.random.rand(n_DoF) + q0span[0]
        X0 = np.array([np.concatenate([p0,q0])])
        
        if model == "FOM":
            X, Xdot = eng.simulation(beam_data,model,DoF,gravity,teval,matlab.double(X0),
                                     disp_progress,nargout=2)
        elif model == "ROM":
            X0 = np.linalg.lstsq(np.array(phi_r), X0.reshape(-1, 2, order='F'), 
                                          rcond=None)[0].reshape(1,-1,order='F'
                                                                 ).squeeze()
            X, Xdot = eng.simulation(beam_data,model,DoF,gravity,teval,matlab.double(X0),
                                     disp_progress,phi_r,nargout=2)
        X = np.array(X)        
        X_data.append(X)
        
        Xdot = np.array(Xdot)        
        Xdot_data.append(Xdot)
             
        t_data.append(tvec.T)
        
        progress_msg = '\rCreating dataset... {:.2f}% done'.format(100*
                                                                   (i+1)/samples)
        print(progress_msg + '\n' if i == samples-1 else progress_msg, end='')
    x_aux = np.concatenate(X_data)
    t = np.zeros_like(x_aux[...,:1])
    data_raw = {'x': np.concatenate(X_data), 
                'dx': np.concatenate(Xdot_data), 
                't': t}
    
    data_raw.update({'n_DoF':n_DoF,'M':M,'I':I,'K':K,'C':C,'phi_r':phi_r})
    if model == 'FOM':
        sio.savemat(beam_data+'_'+model+str(n_DoF)+'_s'+str(samples)+'_'+str(t0)+'-'+str(dt)+'-'+str(t1)+'.mat', data_raw)
    else:
        sio.savemat(beam_data+'_'+model+str(n_modes)+'_s'+str(samples)+'_'+str(t0)+'-'+str(dt)+'-'+str(t1)+'.mat', data_raw)
    # pickle.dump(data_raw, open('dataset'+model+'_'+str(n_DoF)+'.pkl', 'wb'))
    return data_raw
