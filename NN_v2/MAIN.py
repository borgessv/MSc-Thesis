"""
This code is associated to the master's thesis in Aeronautical and Mechanical 
Engineering presented to the Instituto Tecnológico de Aeronáutica and University 
of Twente on July of 2023.

Thesis title: Dynamics of Highly Flexible Slender Beams Using Hamiltonian Neural
Networks
Author: Vitor Borges Santos
Last update: 18-05-2023

Description: 

"""
#%% Loading of required libraries and auxiliary functions
import time, os, sys
import numpy as np
import torch
import matlab.engine
import scipy.io as sio
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', size=11)
plt.rc('axes', labelsize=16) 
plt.rc('xtick', labelsize=11)
plt.rc('xtick', labelsize=11)
from IPython import get_ipython                                                 # force plots to be shown in plot pane
ipython = get_ipython()
ipython.magic('matplotlib inline')

from models import MLP, HNN, DHNN
from data_maker import make_dataset
from train import train, loss_plot
from utils import matlab_interface, ObjectView, pre_process, post_process, meta_model

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
bar = '\\' if 'windows' in sys.platform else '/'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% USER INPUTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_dict = {'dataset': 'beam_brown_ROM1_s20_0-0.001-0.2',                     # options: '.mat' file or 'create'
             'model': 'ROM',                                                    # options: 'FOM' or 'ROM'
             'n_modes': 1,                                                      # number of modes (only for ROM)
             'DoF': {'OutBend'},                                                # options: 'OutBend', 'Inbend', 'Torsion' and/or 'Axial'
             'gravity': 'GravityOn',                                            # options: 'GravityOn' or 'GravityOff'
             'beam_data': 'beam_brown.xlsx',                                # options:  .xlsx file
             'disp_progress': 'False',                                          # options: 'True' or 'False' (must be string)
             'n_samples': 20,
             'train_split': 0.8,
             'tspan': [0, 0.2],
             'timestep': 0.001,
             'p0span': [0, 0],
             'q0span': [-np.pi/9, np.pi/9],
             'normalize': True,
             'norm_range': (0, 1),
             'standardize': False}
nn_dict  =  {'train_model': True,
             'matlab_comparison': True,
             'model': ['MLP', 'HNN', 'DHNN'],                                   # options: 'MLP', 'HNN' and/or 'DHNN'
             'hidden_dim': [32],
             'activation': 'elu',
             'learning_rate': 0.0001,
             'epochs': 1000,
             'batch_size': 64,
             'loss': 'L2',                                                      # options: 'L2' or 'mse'
             'callback': False,
             'device': 'cuda',                                                  # "cpu" or "cuda" (for using GPUs)
             'as_separate': False,
             'decay': 1e-4}
test_dict = {'n_test': 1,
             'tspan': [0, 1],
             't_step': 0.001,
             'p0span': [0, 0],
             'q0span': [-2*np.pi/9, 2*np.pi/9],
             'external_force': False,
             'int_method': 'RK45',
             'rtol': 1e-6,
             'atol': 1e-9,
             'seed': None,
             'animate': False}
###############################################################################
data_args, nn_args, test_args = ObjectView(data_dict), ObjectView(nn_dict), ObjectView(test_dict) 


#%% Dataset pre-processing:
matlab_eng = matlab_interface()
setattr(data_args, 'matlab_engine', matlab_eng) 

if data_args.dataset == 'create':                                             # creates or load dataset:
    data_raw = make_dataset(data_args)
    n_DoF, phi_r = data_raw['n_DoF'], np.array(data_raw['phi_r'])
elif data_args.dataset == 'damped_oscillator':
    from damped_oscillator import dataset, eom, hamiltonian
    mass, k, c = 1, 1, 0.1
    data_raw = dataset(m=mass, k=k, c=c, num_samples=data_args.n_samples, t_span=data_args.tspan, dt=data_args.timestep)
    n_DoF, phi_r = 1, np.array(data_raw['phi_r']) 
else:
    data_raw = sio.loadmat(data_args.dataset)
    n_DoF, phi_r = int(data_raw['n_DoF']), data_raw['phi_r'] 
n = n_DoF if data_args.model == 'FOM' else data_args.n_modes  
data_args.phi_r = matlab.double(phi_r.tolist())

data, x_scaler, dx_scaler = pre_process(data_raw, data_args)

nn_args.input_dim = 2*(data_args.n_modes if data_args.model == 'ROM' else n_DoF)
nn_args.output_dim = 2*(data_args.n_modes if data_args.model == 'ROM' else n_DoF)
nn_args.batch_size = len(data['x_train']) if nn_args.batch_size == 'full' else nn_args.batch_size
nn_args.model.append(data_args.model)

plt.figure('train_data')
for i in range(0,n):
    plt.scatter(data['x_train'][:,n+i], data['x_train'][:,i], color='b',  marker='o', s=5, label='Train') #label='Train: DoF: '+str(i+1)
    plt.scatter(data['x_val'][:,n+i], data['x_val'][:,i], color='r',  marker='o', s=5, label='Validation')

plt.xlabel('$q$ [m]')
plt.ylabel('$p$ [kg$\cdot$m$\cdot$s$^{-1}$]')
plt.grid()
plt.legend(loc='upper right')
plt.savefig('oscillator_data.svg', dpi=300, format='svg', bbox_inches='tight')
plt.show()


#%% Model Training:
model, losses, t_train = {}, {}, {}
for m in nn_args.model[:-1]:
    if nn_args.train_model:
        model[m] = globals()[m](nn_args)
        t = time.time()
        losses[m] = train(model[m], data, nn_args)
        t_train[m] = time.time() - t
        # loss_plot(losses[m])
        model[m].cpu()
        torch.save(model[m].state_dict(), m+'_trained')
    else:
        model[m] = globals()[m](nn_args)
        model[m].load_state_dict(torch.load(m+'_trained'))
        model[m].eval()
        model[m].cpu()


#%% Running Tests
print('\n\rRunning test simulations... {:.2f}% done'.format(0), end='')
np.random.seed(test_args.seed)
setattr(data_args, 'external_force', test_args.external_force)  
tvec = np.linspace(test_args.tspan[0], test_args.tspan[1], 
                   int((test_args.tspan[1]-test_args.tspan[0])/test_args.t_step)+1)
test_sol = {'tvec': tvec}
sol, tsim, tcomp, p, q = {}, {}, {}, {}, {}
for i in range(test_args.n_test):
    p0_test = 0*((test_args.p0span[1]-test_args.p0span[0])*np.random.rand(n_DoF) + test_args.p0span[0])
    q0_test = ((test_args.q0span[1]-test_args.q0span[0])*np.random.rand(n_DoF) + test_args.q0span[0])
    X0_test_raw = np.concatenate([p0_test,q0_test])
    if data_args.model == 'ROM':
        X0_test = np.linalg.lstsq(phi_r, X0_test_raw.reshape(-1, 2, order='F'), 
                                      rcond=None)[0].reshape(1,-1,order='F').squeeze()   
    else:
        X0_test = X0_test_raw
            
    for m in nn_args.model:
        if m == 'FOM' or m == 'ROM':
            if data_args.dataset == 'damped_oscillator':
                tsim[m] = []
                t = time.time()
                sol[m,i] = solve_ivp(lambda t, y: eom(t, y, m=mass, k=k, c=c), test_args.tspan, X0_test, t_eval=tvec)
                tsim[m].append(time.time() - t)
                p[m,i], q[m,i] = post_process(sol[m,i]['y'].T, data_args, phi_r)
            else: 
                tsim[m] = []
                t = time.time()
                sol[m,i] = matlab_eng.simulation(data_args.beam_data, data_args.model, 
                                               data_args.DoF, data_args.gravity,
                                               matlab.double(tvec), matlab.double(X0_test),
                                               data_args.disp_progress, data_args.phi_r, 
                                               nargout=2)[0]
                tsim[m].append(time.time() - t)
                sol[m,i] = np.array(sol[m,i])
                p[m,i], q[m,i] = post_process(sol[m,i], data_args, phi_r)
        else:
            tsim[m] = []
            t = time.time()
            args = (model[m], x_scaler, dx_scaler, data_args)
            kwargs = {'t_eval': tvec, 'rtol':test_args.rtol, 'atol':test_args.atol}
            sol[m,i] = solve_ivp(meta_model, test_args.tspan, X0_test.flatten(), 
                               args=args, method=test_args.int_method, **kwargs)
            tsim[m].append(time.time() - t)
            p[m,i], q[m,i] = post_process(sol[m,i]['y'].T, data_args, phi_r)
    
    progress_msg = '\rRunning test simulations... {:.2f}% done'.format(100*(i+1)/test_args.n_test)
    print(progress_msg + '\n' if i == test_args.n_test-1 else progress_msg, end='')

for m in nn_args.model[:-1]:
    tcomp[m] = (np.array(tsim[m])/np.array(tsim[nn_args.model[-1]]))*100
    print('\n{} to FOM/ROM relative computational cost: {:.2f}% +/- {:.2f}%'.format(m, np.mean(tcomp[m]),np.std(tcomp[m])))


#%% Plotting Results
test_num = 0
cmap = plt.cm.tab10
if data_args.dataset != 'damped_oscillator':
    for m in nn_args.model:
        tip_pos, tip_ang = matlab_eng.tip_position(data_args.beam_data,matlab.double(q[m,test_num].tolist()), data_args.DoF, nargout=2)
        tip_pos, tip_ang = np.array(tip_pos), np.array(tip_ang)
        plt.figure(1)
        plt.plot(tvec, tip_pos[:,0], label=m+' (n='+str(len(X0_test)//2)+')' if m == 'FOM' or m == 'ROM' else m)
        plt.figure(2)
        plt.plot(tvec, tip_pos[:,1], label=m+' (n='+str(len(X0_test)//2)+')' if m == 'FOM' or m == 'ROM' else m)
        plt.figure(3)
        plt.plot(tvec, tip_pos[:,2], label=m+' (n='+str(len(X0_test)//2)+')' if m == 'FOM' or m == 'ROM' else m)
        if 'Torsion' in data_args.DoF:
            plt.figure(4)
            plt.plot(tvec, tip_ang, label=m+' (n='+str(len(X0_test)//2)+')' if m == 'FOM' or m == 'ROM' else m)
    
    plt.figure(1)
    plt.xlabel('time [s]')
    plt.ylabel('$x_{\mathrm{tip}}$ [m]')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.figure(2)
    plt.xlabel('time [s]')
    plt.ylabel('$y_{\mathrm{tip}}$ [m]')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.figure(3)
    plt.xlabel('time [s]')
    plt.ylabel('$z_{\mathrm{tip}}$ [m]')
    plt.grid(True)
    plt.legend(loc='upper right')
    if 'Torsion' in data_args.DoF:
        plt.figure(4)
        plt.xlabel('time [s]')
        plt.ylabel('$\\theta_{\mathrm{tip}}$ [deg]')
        plt.grid(True)
        plt.legend(loc='upper right')
    plt.show()
    
    if test_args.animate:
        matlab_eng.animate(data_args.beam_data, "test_NN.mat", nn_args.model, 
                            data_args.DoF, n, 'gif', nargout=0)
    
    if 'matlab_eng' in locals():
        matlab_eng.quit()
        del matlab_eng

else:
    for i, m in enumerate(nn_args.model):
        plt.figure(1)
        plt.plot(q[m, test_num], p[m, test_num], color='k' if m == 'FOM' or m == 'ROM' else cmap(i), label='Ground Truth' if m == 'FOM' or m == 'ROM' else m) #label=m+' (n='+str(len(X0_test)//2)+')' if m == 'FOM' or m == 'ROM' else m
        plt.figure(2)
        plt.plot(tvec, q[m, test_num], color='k' if m == 'FOM' or m == 'ROM' else cmap(i), linestyle='--' if m == 'FOM' or m == 'ROM' else '-', label='Ground Truth' if m == 'FOM' or m == 'ROM' else m) 
    plt.figure(1)
    plt.xlabel('$q$ [m]')
    plt.ylabel('$p$ [kg$\cdot$m$\cdot$s$^{-1}$]')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig('oscillator_sol.svg', dpi=300, format='svg', bbox_inches='tight')
    plt.figure(2)
    plt.xlabel('Time [s]')
    plt.ylabel('$q$ [m]')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig('oscillator_qsol.svg', dpi=300, format='svg', bbox_inches='tight')
    plt.show()
    
    H = hamiltonian(p['FOM', 0], q['FOM', 0], m=mass, k=k, c=c)
    H_MLP = hamiltonian(p['MLP', 0], q['MLP', 0], m=mass, k=k, c=c)
    H_HNN = hamiltonian(p['HNN', 0], q['HNN', 0], m=mass, k=k, c=c)
    H_DHNN = hamiltonian(p['DHNN', 0], q['DHNN', 0], m=mass, k=k, c=c)
    x = sol['HNN',0]['y'].T
    x = torch.tensor(x, requires_grad=True, dtype=torch.float32, device=torch.device('cpu'))
    H_hat = model['HNN'](x)['H_hat']
    H_hat = H_hat.data.numpy()
    fig, ax = plt.subplots()
    
    # plt.plot(tvec, H_hat, color='r', label='HNN')
    plt.plot(tvec, H_MLP, color='g', label='MLP')
    plt.plot(tvec, H_HNN, color='b', label='HNN')
    plt.plot(tvec, H_DHNN, color='r', label='DHNN')
    plt.plot(tvec, H, '-k', label='Ground Truth')
    # plt.plot(tvec, D_hat, color='r', label='$\hat{D}$')
    plt.ylabel('Total Energy [J]')
    plt.xlabel('Time [s]')
    ax.legend(loc='upper right')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.grid(which='both')
    plt.savefig('oscillator_H.svg', dpi=300, format='svg', bbox_inches='tight')
    plt.show()
#%%
# # Define the optimizer and loss function
# optimizer = torch.optim.Adam(model.parameters(), lr=nn_args.learning_rate)
# criterion = torch.nn.MSELoss()

# # Convert the data to PyTorch tensors if not already
# model.to(device)
# x_train = torch.tensor(data['x_train'], dtype=torch.float32, requires_grad=True, device=device)
# dx_train = torch.tensor(data['dx_train'], dtype=torch.float32, device=device)
# x_val = torch.tensor(data['x_val'], dtype=torch.float32, requires_grad=True, device=device)
# dx_val = torch.tensor(data['dx_val'], dtype=torch.float32, device=device)

# train_losses = []
# val_losses = []
# # Train the model
# for epoch in range(nn_args.epochs):
#     # Set the model to training mode
#     model.train()
    
#     indices = torch.randperm(len(x_train))
#     x_train_shuffled = x_train[indices]
#     dx_train_shuffled = dx_train[indices]
#     x_val_shuffled = x_train[indices]
#     dx_val_shuffled = dx_train[indices]
    
#     for i in range(0, len(x_train_shuffled), nn_args.batch_size):
#         # Get mini-batch
#         x_batch = x_train_shuffled[i:i+nn_args.batch_size]
#         dx_batch = dx_train_shuffled[i:i+nn_args.batch_size]
#         x_batch_val = x_val_shuffled[i:i+nn_args.batch_size]
#         dx_batch_val = dx_val_shuffled[i:i+nn_args.batch_size]
        
#         # Forward pass
#         Xdot_hat = model(x_batch)['x_hat']
#         loss = criterion(Xdot_hat, dx_batch)
        
#         # Backward pass and optimization
        
#         loss.backward(retain_graph=True)
#         optimizer.step()
#         optimizer.zero_grad(set_to_none=True)
#     # # Forward pass
#     # dx_hat = model(x_train)
#     # loss = (dx_train-dx_hat).pow(2).mean()
    
#     # # Backward pass and optimization
#     # optimizer.zero_grad()
#     # loss.backward()
#     # optimizer.step()
    
#     # Set the model to evaluation mode
#         model.eval()
        
#         # Calculate validation loss
#         # with torch.no_grad():
#         Xdot_hat_val = model(x_val)['x_hat']
#         val_loss = criterion(Xdot_hat_val, dx_val)
    
#     train_losses.append(loss.item())
#     val_losses.append(val_loss.item())
    
#     # Print progress
#     print(f"Epoch {epoch+1}/{nn_args.epochs}, Loss: {loss.item():.6e}, Val Loss: {val_loss.item():.6e}")
#%%
# model = model.cpu()

# fig, ax = plt.subplots()
# plt.plot(train_losses, color='b', label='Train')
# plt.plot(val_losses, color='r', label='Validation')
# ax.set_yscale('log') 
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# ax.legend(loc='upper right')
# ax.grid(which='both')
# plt.show()


#%%
# model = MLP(nn_args)
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=nn_args.learning_rate), 
#               loss=tf.keras.losses.MeanSquaredError())

# # Train the model
# model.fit(data['x_train'], data['dx_train'], 
#           validation_data=(data['x_val'], data['dx_val']), 
#           epochs=nn_args.epochs, batch_size=nn_args.batch_size, 
#           callbacks=[PlotCallback(nn_args.epochs) if nn_args.callback else []])

#%%
# # if args.train_model is True:
# #     if 'DHNN_model' in locals(): del DHNN_model
# #     if 'HNN_model' in locals(): del HNN_model
# #     if 'MLP_model' in locals(): del MLP_model
# if args.train_model:
#     for nn in args.model:
#         if 'nn'+'_model' in locals(): del locals()[nn+'_model']
#     if 'MLP' in args.model:
#         MLP_model = MLP(args.input_dim, args.output_dim, args.hidden_dim)
#         MLP_results = train(MLP_model, args, data)
#         torch.save(MLP_model.state_dict(), 'MLP_trained')
#         MLP_model.eval()
#     if 'HNN' in args.model:
#         HNN_model = HNN(args.input_dim, args.hidden_dim)
#         HNN_results = train(HNN_model, args, data)
#         torch.save(HNN_model.state_dict(), 'HNN_trained')
#         HNN_model.eval()
#     if 'DHNN' in args.model:
#         DHNN_model = DHNN(args.input_dim, args.hidden_dim)
#         DHNN_results = train(DHNN_model, args, data)
#         torch.save(DHNN_model.state_dict(), 'DHNN_trained')
#         DHNN_model.eval()
# else:
#     if 'MLP' in args.model:
#         MLP_model.load_state_dict(torch.load('MLP_trained'))
#         MLP_model.eval()
#     if 'HNN' in args.model:
#         HNN_model.load_state_dict(torch.load('HNN_trained'))
#         HNN_model.eval()
#     if 'DHNN' in args.model:
#         DHNN_model.load_state_dict(torch.load('DHNN_trained'))
#         DHNN_model.eval()

#%% Test simulation model:
# print('\rRunning test simulations... {:.2f}% done'.format(0), end='')




# tsim_mlp, tsim_hnn, tsim_dhnn, tsim_true, t_mlp_true, t_hnn_true, t_dhnn_true = [], [], [], [], [], [], []
# test_sol = {'tvec': tvec}
# sol, tsim, p, q = {}, {}, {}, {}
# for i in range(test_args.n_test):
#     p0_test = 0*((test_args.p0span[1]-test_args.p0span[0])*np.random.rand(n_DoF) + test_args.p0span[0])
#     q0_test = 0*((test_args.q0span[1]-test_args.q0span[0])*np.random.rand(n_DoF) + test_args.q0span[0])
#     X0_test_raw = np.concatenate([p0_test,q0_test])
#     if data_args.model == 'ROM':
#         X0_test = np.linalg.lstsq(phi_r, X0_test_raw.reshape(-1, 2, order='F'), 
#                                       rcond=None)[0].reshape(1,-1,order='F').squeeze()
         
#     for m in nn_args.model:
#         if m == 'FOM' or m == 'ROM':
#             t = time.time()
#             sol[m,i] = matlab_eng.simulation(data_args.beam_data, data_args.model, 
#                                            data_args.DoF, data_args.gravity,
#                                            matlab.double(tvec), matlab.double(X0_test),
#                                            data_args.disp_progress, data_args.phi_r, 
#                                            nargout=2)[0]
#             tsim[m,i].append(time.time() - t)
#             sol[m,i] = np.array(sol[m])
#             p[m,i], q[m,i] = post_process(sol[m], data_args, phi_r)
#         else:
#             t = time.time()
#             args = (model[m], x_scaler, dx_scaler, matlab_eng, data_args)
#             kwargs = {'t_eval': tvec, 'atol': test_args.atol, 'rtol': test_args.rtol}
#             sol[m,i] = solve_ivp(meta_dynamics, test_args.tspan, X0_test.flatten(), 
#                                args=args, method=test_args.method, **kwargs)
#             tsim[m,i].append(time.time() - t)
#             p[m,i], q[m,i] = post_process(sol[m,i]['y'].T, data_args, phi_r)
    
#     progress_msg = '\rrunning test simulations... {:.2f}% done'.format(100*(i+1)/test_args.n_test)
            
#     # MLP_sol = solve_ivp(dynamics, tspan, X0_test, t_eval=tvec)
#     if 'DHNN' in nn_args.model:                                                  # runs MLP model simulation for the test case
#         t = time.time()
#         sol[m] = solve_ivp(meta_dynamics, tspan, X0_test.flatten(), args=(model, x_scaler, dx_scaler, matlab_eng, data_args), **kwargs)
        
#         # MLP_sol = integrate_model(dynamics, tspan, X0_test, **kwargs) 
#         tsim_mlp.append(time.time() - t)
    
#     # x = x_scaler.transform(MLP_sol['y'].T)
#     # x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
#     # H_hat, D_hat = model(x)['H_hat'], model(x)['D_hat']
#     # H_hat, D_hat = H_hat.data.numpy(), D_hat.data.numpy()
#     # fig, ax = plt.subplots()
#     # plt.plot(tvec, H_hat, color='b', label='$\hat{H}$')
#     # plt.plot(tvec, D_hat, color='r', label='$\hat{D}$')
#     # plt.xlabel('Epoch')
#     # plt.ylabel('Loss')
#     # ax.legend(loc='upper right')
#     # ax.grid(which='both')
#     # plt.show()
    
#     # if 'HNN' in args.model:                                                  # runs HNN model simulation for the test case
#     #     t = time.time()
#     #     HNN_sol = integrate_model(HNN_model, tspan, X0_test, **kwargs) 
#     #     tsim_hnn.append(time.time() - t)
#     # if 'DHNN' in args.model:                                                 # runs DHNN model simulation for the test case
#     #     t = time.time()
#     #     DHNN_sol = integrate_model(DHNN_model, tspan, X0_test, **kwargs) 
#     #     tsim_dhnn.append(time.time() - t)
 
#     if nn_args.matlab_comparison:                                                  # runs FOM/ROM model for the test case
#         if not 'matlab_eng' in locals():
#             # import matlab.engine
#             matlab_eng = matlab_interface()
#             setattr(data_args, 'matlab_engine', matlab_eng)                                    
        
#         # p_true, q_true = post_process(X_true, data_raw, data_args) if data_args.model == 'FOM' else post_process(X_true, data_raw, data_args, phi_r) 
#         if data_args.model == 'FOM':
#             p_true, q_true = np.split(X_true, 2, axis=1)
#             test_sol.update({'X_FOM': np.concatenate([p_true,q_true])})
#         elif data_args.model == 'ROM':
#             etap_true, eta_true = np.split(X_true, 2, axis=1)
#             p_true, q_true = (phi_r@etap_true.T).T, (phi_r@eta_true.T).T
#             test_sol.update({'X_ROM': np.concatenate([p_true,q_true])})
        
#         if 'MLP' in nn_args.model:                                              # NN model relative computaional cost:
#             t_mlp_true.append((tsim_mlp[i]/tsim_true[i])*100)
#         # if 'HNN' in args.model:
#         #     t_hnn_true.append((tsim_hnn[i]/tsim_true[i])*100)
#         # if 'DHNN' in args.model:
#         #     t_dhnn_true.append((tsim_dhnn[i]/tsim_true[i])*100)
                
#     progress_msg = '\rrunning test simulations... {:.2f}% done'.format(100*(i+1)/n_test)
#     print(progress_msg + '\n' if i == n_test-1 else progress_msg, end='')
  
# if 'DHNN' in nn_args.model: 
#     p_MLP, q_MLP = post_process(MLP_sol['y'].T, dx_scaler, data_raw, data_args) if data_args.model == 'FOM' else post_process(MLP_sol['y'].T, x_scaler, data_raw, data_args, phi_r) 
#     test_sol.update({'X_MLP': np.concatenate([p_MLP,q_MLP])})
#     if nn_args.matlab_comparison:
#         print('\nMLP to FOM/ROM relative computational cost: {:.2f}% +/- {:.2f}%'.format(np.mean(t_mlp_true),np.std(t_mlp_true)))
# # if 'HNN' in args.model: 
#     p_HNN, q_HNN = post_process(HNN_sol['y'].T, data_raw, data_args) if data_args.model == 'FOM' else post_process(HNN_sol['y'].T, data_raw, data_args, phi_r) 
#     test_sol.update({'X_HNN': np.concatenate([p_HNN,q_HNN])})
#     if args.matlab_comparison:
#         print('\nHNN to FOM/ROM relative computational cost: {:.2f}% +/- {:.2f}%'.format(np.mean(t_hnn_true),np.std(t_hnn_true)))
# if 'DHNN' in args.model: 
#     p_DHNN, q_DHNN = post_process(DHNN_sol['y'].T, data_raw, data_args) if data_args.model == 'FOM' else post_process(DHNN_sol['y'].T, data_raw, data_args, phi_r) 
#     test_sol.update({'X_DHNN': np.concatenate([p_DHNN,q_DHNN])})
#     if args.matlab_comparison:
#         print('\nD-HNN to FOM/ROM relative computational cost: {:.2f}% +/- {:.2f}%'.format(np.mean(t_dhnn_true),np.std(t_dhnn_true)))
sio.savemat(parent_dir + bar + "SimulationFramework" + bar + "test_results" + bar + "test_NN.mat", test_sol)

#%% Tip position plot:

# plt.rc('text', usetex=True)
# plt.rc('font', size=11)
# plt.rc('axes', labelsize=16) 
# plt.rc('xtick', labelsize=11)
# plt.rc('xtick', labelsize=11)

# test_number = 0
# for m in nn_args.model:
#     tip_pos, tip_ang = matlab_eng.tip_position(data_args.beam_data,matlab.double(q[m,test_number].tolist()), data_args.DoF, nargout=2)
#     tip_pos, tip_ang = np.array(tip_pos), np.array(tip_ang)
#     plt.figure(1)
#     plt.plot(tvec, tip_pos[:,0], label=m+' (n='+str(len(X0_test)//2)+')' if m == 'FOM' or m == 'ROM' else m)
#     plt.figure(2)
#     plt.plot(tvec, tip_pos[:,1], label=m+' (n='+str(len(X0_test)//2)+')' if m == 'FOM' or m == 'ROM' else m)
#     plt.figure(3)
#     plt.plot(tvec, tip_pos[:,2], label=m+' (n='+str(len(X0_test)//2)+')' if m == 'FOM' or m == 'ROM' else m)
#     if 'Torsion' in data_args.DoF:
#         plt.figure(4)
#         plt.plot(tvec, tip_ang, label=m+' (n='+str(len(X0_test)//2)+')' if m == 'FOM' or m == 'ROM' else m)

# plt.figure(1)
# plt.xlabel('time [s]')
# plt.ylabel('$x_{\mathrm{tip}}$ [m]')
# plt.grid(True)
# plt.legend(loc='upper right')
# # plt.show()
# plt.figure(2)
# plt.xlabel('time [s]')
# plt.ylabel('$y_{\mathrm{tip}}$ [m]')
# plt.grid(True)
# plt.legend(loc='upper right')
# # plt.show()
# plt.figure(3)
# plt.xlabel('time [s]')
# plt.ylabel('$z_{\mathrm{tip}}$ [m]')
# plt.grid(True)
# plt.legend(loc='upper right')

# plt.figure(4)
# plt.xlabel('time [s]')
# plt.ylabel('$\\theta_{\mathrm{tip}}$ [deg]')
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.show()
# plt.show()
#%%
# if nn_args.matlab_comparison:
#     tip_true,ang_tip = matlab_eng.tip_position(data_args.beam_data,matlab.double(q_true.tolist()), data_args.DoF, nargout=2)
#     tip_true,ang_tip = np.array(tip_true), np.array(ang_tip)
#     plt.figure(1)
#     plt.plot(tvec, tip_true[:,0], 'm', label='x: '+data_args.model+' (n='+str(int(X0_test.shape[1]/2))+')')
#     plt.figure(2)
#     plt.plot(tvec, tip_true[:,1], 'k', label='y: '+data_args.model+' (n='+str(int(X0_test.shape[1]/2))+')')
#     plt.figure(3)
#     plt.plot(tvec, tip_true[:,2], 'c', label='z: '+data_args.model+' (n='+str(int(X0_test.shape[1]/2))+')')
# if 'DHNN' in nn_args.model: 
#     tip_MLP,ang_MLP = matlab_eng.tip_position(data_args.beam_data,matlab.double(q_MLP.tolist()), data_args.DoF, nargout=2)
#     tip_MLP,ang_MLP = np.array(tip_MLP), np.array(ang_MLP)
#     plt.figure(1)
#     plt.plot(tvec, tip_MLP[:,0], ':r', label='x: MLP')
#     plt.figure(2)
#     plt.plot(tvec, tip_MLP[:,1], ':g', label='y: MLP')
#     plt.figure(3)
#     plt.plot(tvec, tip_MLP[:,2], ':b', label='z: MLP')
# if 'HNN' in args.model: 
#     tip_HNN = np.array(matlab_eng.tip_position(data_args.beam_data,matlab.double(q_HNN.tolist()), data_args.DoF, nargout=1))
#     plt.plot(tvec, tip_HNN[:,0], '-.r', label='x: HNN')
#     plt.plot(tvec, tip_HNN[:,1], '-.g', label='y: HNN')
#     plt.plot(tvec, tip_HNN[:,2], '-.b', label='z: HNN')
# if 'DHNN' in args.model: 
#     tip_DHNN = np.array(matlab_eng.tip_position(data_args.beam_data,matlab.double(q_DHNN.tolist()), data_args.DoF, nargout=1))
#     plt.plot(tvec, tip_DHNN[:,0], '--r', label='x: D-HNN')
#     plt.plot(tvec, tip_DHNN[:,1], '--g', label='y: D-HNN')
#     plt.plot(tvec, tip_DHNN[:,2], '--b', label='z: D-HNN')

# plt.figure(1)
# plt.xlabel('time [s]')
# plt.ylabel('$x_{\mathrm{tip}}$ [m]')
# plt.grid(True)
# plt.legend(loc='upper right')
# # plt.show()
# plt.figure(2)
# plt.xlabel('time [s]')
# plt.ylabel('$y_{\mathrm{tip}}$ [m]')
# plt.grid(True)
# plt.legend(loc='upper right')
# # plt.show()
# plt.figure(3)
# plt.xlabel('time [s]')
# plt.ylabel('$z_{\mathrm{tip}}$ [m]')
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.show()


# # plt.xlabel('time [s]')
# # plt.ylabel('tip position [m]')
# # plt.grid(True)
# # plt.legend(loc='upper right')
# # plt.show()

# if 'Torsion' in data_args.DoF:
#     plt.figure(4)
#     if nn_args.matlab_comparison:
#         plt.plot(tvec, ang_tip, 'k', label=data_args.model+' (n='+str(int(X0_test.shape[1]/2))+')')
#     if 'DHNN' in nn_args.model: 
#         plt.plot(tvec, ang_MLP[:,0], ':g', label='MLP')
# # if 'HNN' in args.model: 
# #     tip_HNN = np.array(matlab_eng.tip_position(data_args.beam_data,matlab.double(q_HNN.tolist()), data_args.DoF, nargout=1))
# #     plt.plot(tvec, tip_HNN[:,0], '-.r', label='x: HNN')
# #     plt.plot(tvec, tip_HNN[:,1], '-.g', label='y: HNN')
# #     plt.plot(tvec, tip_HNN[:,2], '-.b', label='z: HNN')
# # if 'DHNN' in args.model: 
# #     tip_DHNN = np.array(matlab_eng.tip_position(data_args.beam_data,matlab.double(q_DHNN.tolist()), data_args.DoF, nargout=1))
# #     plt.plot(tvec, tip_DHNN[:,0], '--r', label='x: D-HNN')
# #     plt.plot(tvec, tip_DHNN[:,1], '--g', label='y: D-HNN')
# #     plt.plot(tvec, tip_DHNN[:,2], '--b', label='z: D-HNN')
#     plt.xlabel('time [s]')
#     plt.ylabel('$\\theta_{\mathrm{tip}}$ [deg]')
#     plt.grid(True)
#     plt.legend(loc='upper right')
#     plt.show()

# #%% Animation:
# # matlab_eng.animate(data_args.beam_data,"test_NN.mat", nn_args.model, 
# #                     data_args.DoF, n, 'gif', nargout=0)

# if 'matlab_eng' in locals():
#     matlab_eng.quit()
#     del matlab_eng
    
#%% Phase portrait for last test run:
# DoF = 0                                                                         # DoF considered for the phase portrait 
# ipython.magic('matplotlib inline')
# fig = plt.figure()
# if args.matlab_comparison:
#     # p_true, q_true = np.split(X_true, 2, axis=1)
#     plt.plot(q_true[:,DoF],p_true[:,DoF], '--k', label='Ground Truth')
# if 'MLP' in args.model: 
#     # p_MLP, q_MLP = np.split(X_MLP, 2, axis=1)
#     plt.plot(q_MLP[:,DoF],p_MLP[:,DoF], 'g', label='MLP')
# if 'HNN' in args.model: 
#     # p_HNN, q_HNN = np.split(X_HNN, 2, axis=1)
#     plt.plot(q_HNN[:,DoF],p_HNN[:,DoF], 'b', label='HNN')
# # if 'DHNN' in args.model: 
#     # p_DHNN, q_DHNN = np.split(X_DHNN, 2, axis=1)
#     # plt.plot(q_DHNN[:,DoF],p_DHNN[:,DoF], 'r', label='D-HNN')
# plt.xlabel('q [rad]')
# plt.ylabel('p [kg*rad/s]')
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.show()