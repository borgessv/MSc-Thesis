# -*- coding: utf-8 -*-
"""
"""
import time, os, sys
import numpy as np
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
from IPython import get_ipython                                                 # force plots to be shown in plot pane
ipython = get_ipython()

from models import MLP, HNN, DHNN
from train import train
from data_maker import make_dataset
from utils import matlab_interface, ObjectView, normalize, denormalize, standardize, destandardize, integrate_model, post_process

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
if 'windows' in sys.platform:
    bar = '\\'
else:
    bar = '/'

#%% Simulation model and dataset parameters (USER INPUT):
def model_args(as_dict=False):
  model_dict = {'dataset': 'hfw_patil.xlsx_ROM2_s100_0-0.001-1',                                            # '.mat' file or 'create'
                'model': 'ROM',                                                 # 'FOM' or 'ROM'
                'n_modes': 2,                                                   # only for 'ROM'
                'DoF': {'OutBend'},                                             # 'OutBend', 'Inbend', 'Torsion' or 'Axial'
                'gravity': 'GravityOn',                                         # 'GravityOn' or 'GravityOff'
                'beam_data': 'hfw_patil.xlsx',                             # .xlsx file
                'disp_progress': 'False',                                       # 'True' or 'False' (must be string)
                'n_samples': 100,
                'train_split': 0.8,
                'tspan': [0, 0.5],
                'timestep': 0.001,
                'p0span': [0, 0],
                'q0span': [-np.pi/9, np.pi/9],
                'normalize': False,
                'norm_range': [0, 1],
                'standardize': False}
  return model_dict if as_dict else ObjectView(model_dict)
model_param = model_args()

if model_param.dataset == 'create':                                             # creates or load dataset:
    import matlab.engine
    matlab_eng = matlab_interface()
    setattr(model_param, 'matlab_engine', matlab_eng)
    data_raw = make_dataset(model_param)
    n_DoF, phi_r = data_raw['n_DoF'], np.array(data_raw['phi_r']) 
else:
    data_raw = sio.loadmat(model_param.dataset)
    n_DoF, phi_r = int(data_raw['n_DoF']), data_raw['phi_r'] 
    # data_raw = pickle.load(open(model_param.dataset, 'rb'))
n = n_DoF if model_param.model == 'FOM' else model_param.n_modes                             
#%% Pre-processing dataset:
data = {}
data['t'] = data_raw['t']
if model_param.normalize:                                                       # normalizes/standardizes dataset (if enabled)
    for k in ['x', 'dx']:
        data[k] = normalize(data_raw[k], data_raw[k], model_param.norm_range)
elif model_param.standardize:
    for k in ['x', 'dx']:
        data[k] = standardize(data_raw[k], data_raw[k])
else:
    for k in ['x', 'dx']:
        data[k] = data_raw[k]

n_train = round(model_param.n_samples*model_param.train_split)                  # train/test split:
split_index = int(n_train*len(data['x'])/model_param.n_samples)
split_data = {}
for k in ['x', 'dx','t']:
    split_data[k], split_data[k + '_test'] = data[k][:split_index], data[k][split_index:]
data = split_data

#%% Train NN model:
def train_args(as_dict=False):                                                  # train and test parameters (USER INPUT)
  train_dict = {'train_model': True,
                'matlab_comparison': True,
                'ml_model': ['MLP'],                                     # 'MLP', 'HNN' or 'DHNN'
                'input_dim': 1 + 2*(model_param.n_modes if model_param.model == 'ROM' else n_DoF),
                'hidden_dim': 12, 
                'output_dim': 2*(model_param.n_modes if model_param.model == 'ROM' else n_DoF),
                'learning_rate': 5e-4, 
                'test_every': 50,
                'print_every': 200,
                'batch_size': split_index,
                'total_steps': 70000,  
                'device': 'cpu',                                                # "cpu" or "cuda" (for using GPUs)
                'seed': None,
                'as_separate': False,
                'decay': 1e-4}
  return train_dict if as_dict else ObjectView(train_dict)
args = train_args()
args.ml_model.insert(0,model_param.model)

# if args.train_model is True:
#     if 'DHNN_model' in locals(): del DHNN_model
#     if 'HNN_model' in locals(): del HNN_model
#     if 'MLP_model' in locals(): del MLP_model
if args.train_model:
    for nn in args.ml_model:
        if 'nn'+'_model' in locals(): del locals()[nn+'_model']
    if 'MLP' in args.ml_model:
        MLP_model = MLP(args.input_dim, args.output_dim, args.hidden_dim)
        MLP_results = train(MLP_model, args, data)
        torch.save(MLP_model.state_dict(), 'MLP_trained')
        MLP_model.eval()
    if 'HNN' in args.ml_model:
        HNN_model = HNN(args.input_dim, args.hidden_dim)
        HNN_results = train(HNN_model, args, data)
        torch.save(HNN_model.state_dict(), 'HNN_trained')
        HNN_model.eval()
    if 'DHNN' in args.ml_model:
        DHNN_model = DHNN(args.input_dim, args.hidden_dim)
        DHNN_results = train(DHNN_model, args, data)
        torch.save(DHNN_model.state_dict(), 'DHNN_trained')
        DHNN_model.eval()
else:
    if 'MLP' in args.ml_model:
        MLP_model.load_state_dict(torch.load('MLP_trained'))
        MLP_model.eval()
    if 'HNN' in args.ml_model:
        HNN_model.load_state_dict(torch.load('HNN_trained'))
        HNN_model.eval()
    if 'DHNN' in args.ml_model:
        DHNN_model.load_state_dict(torch.load('DHNN_trained'))
        DHNN_model.eval()

#%% Test simulation model:
print('\rRunning test simulations... {:.2f}% done'.format(0), end='')

np.random.seed(args.seed)  
tspan = [0,15]                                                                  # defines test parameters
dt = 0.1
tvec = np.array(np.linspace(tspan[0],tspan[1],int((tspan[1]-tspan[0])/dt)+1))
kwargs = {'t_eval': tvec, 'rtol': 1e-8}
p0span = model_param.p0span
q0span = model_param.q0span
n_test = 1

tsim_mlp, tsim_hnn, tsim_dhnn, tsim_true, t_mlp_true, t_hnn_true, t_dhnn_true = [], [], [], [], [], [], []
test_sol = {'tvec': tvec}
for i in range(n_test):
    p0_test = 0*((p0span[1]-p0span[0])*np.random.rand(n_DoF) + p0span[0])
    q0_test = 2*((q0span[1]-q0span[0])*np.random.rand(n_DoF) + q0span[0])
    X0_test_raw = np.concatenate([p0_test,q0_test])
    if model_param.model == 'ROM':
        X0_test_raw = np.linalg.lstsq(phi_r, X0_test_raw.reshape(-1, 2, order='F'), 
                                      rcond=None)[0].reshape(1,-1,order='F').squeeze()
        
    if model_param.normalize:
        X0_test = normalize(data_raw['x'], X0_test_raw.reshape(1,-1), model_param.norm_range).squeeze()
    elif model_param.standardize:
        X0_test = standardize(data_raw['x'], X0_test_raw.reshape(1,-1)).squeeze()
    else:
        X0_test = X0_test_raw   
   
    if 'MLP' in args.ml_model:                                                  # runs MLP model simulation for the test case
        t = time.time()
        MLP_sol = integrate_model(MLP_model, tspan, X0_test, **kwargs) 
        tsim_mlp.append(time.time() - t)
    if 'HNN' in args.ml_model:                                                  # runs HNN model simulation for the test case
        t = time.time()
        HNN_sol = integrate_model(HNN_model, tspan, X0_test, **kwargs) 
        tsim_hnn.append(time.time() - t)
    if 'DHNN' in args.ml_model:                                                 # runs DHNN model simulation for the test case
        t = time.time()
        DHNN_sol = integrate_model(DHNN_model, tspan, X0_test, **kwargs) 
        tsim_dhnn.append(time.time() - t)
    
    if args.matlab_comparison:                                                  # runs FOM/ROM model for the test case
        if not 'matlab_eng' in locals():
            import matlab.engine
            matlab_eng = matlab_interface()
            setattr(model_param, 'matlab_engine', matlab_eng)                                    
        t = time.time()
        X_true = matlab_eng.simulation(model_param.beam_data,model_param.model,model_param.DoF,model_param.gravity,
                                       matlab.double(tvec),matlab.double(X0_test_raw),
                                       model_param.disp_progress,matlab.double(phi_r.tolist()),nargout=2)[0]
        tsim_true.append(time.time() - t)
        X_true = np.array(X_true)
        # p_true, q_true = post_process(X_true, data_raw, model_param) if model_param.model == 'FOM' else post_process(X_true, data_raw, model_param, phi_r) 
        if model_param.model == 'FOM':
            p_true, q_true = np.split(X_true, 2, axis=1)
            test_sol.update({'X_FOM': np.concatenate([p_true,q_true])})
        elif model_param.model == 'ROM':
            etap_true, eta_true = np.split(X_true, 2, axis=1)
            p_true, q_true = (phi_r@etap_true.T).T, (phi_r@eta_true.T).T
            test_sol.update({'X_ROM': np.concatenate([p_true,q_true])})
        
        if 'MLP' in args.ml_model:                                              # NN model relative computaional cost:
            t_mlp_true.append((tsim_mlp[i]/tsim_true[i])*100)
        if 'HNN' in args.ml_model:
            t_hnn_true.append((tsim_hnn[i]/tsim_true[i])*100)
        if 'DHNN' in args.ml_model:
            t_dhnn_true.append((tsim_dhnn[i]/tsim_true[i])*100)
                
    progress_msg = '\rrunning test simulations... {:.2f}% done'.format(100*(i+1)/n_test)
    print(progress_msg + '\n' if i == n_test-1 else progress_msg, end='')
  
if 'MLP' in args.ml_model: 
    p_MLP, q_MLP = post_process(MLP_sol['y'].T, data_raw, model_param) if model_param.model == 'FOM' else post_process(MLP_sol['y'].T, data_raw, model_param, phi_r) 
    test_sol.update({'X_MLP': np.concatenate([p_MLP,q_MLP])})
    if args.matlab_comparison:
        print('\nMLP to FOM/ROM relative computational cost: {:.2f}% +/- {:.2f}%'.format(np.mean(t_mlp_true),np.std(t_mlp_true)))
if 'HNN' in args.ml_model: 
    p_HNN, q_HNN = post_process(HNN_sol['y'].T, data_raw, model_param) if model_param.model == 'FOM' else post_process(HNN_sol['y'].T, data_raw, model_param, phi_r) 
    test_sol.update({'X_HNN': np.concatenate([p_HNN,q_HNN])})
    if args.matlab_comparison:
        print('\nHNN to FOM/ROM relative computational cost: {:.2f}% +/- {:.2f}%'.format(np.mean(t_hnn_true),np.std(t_hnn_true)))
if 'DHNN' in args.ml_model: 
    p_DHNN, q_DHNN = post_process(DHNN_sol['y'].T, data_raw, model_param) if model_param.model == 'FOM' else post_process(DHNN_sol['y'].T, data_raw, model_param, phi_r) 
    test_sol.update({'X_DHNN': np.concatenate([p_DHNN,q_DHNN])})
    if args.matlab_comparison:
        print('\nD-HNN to FOM/ROM relative computational cost: {:.2f}% +/- {:.2f}%'.format(np.mean(t_dhnn_true),np.std(t_dhnn_true)))
sio.savemat(parent_dir + bar + "SimulationFramework" + bar + "test_results" + bar + "test_NN.mat", test_sol)

#%% Tip position plot:
ipython.magic('matplotlib inline')
fig = plt.figure()
if args.matlab_comparison:
    tip_true = np.array(matlab_eng.tip_position(model_param.beam_data,matlab.double(q_true.tolist()), model_param.DoF, nargout=1))
    plt.plot(tvec, tip_true[:,0], 'm', label='x: '+model_param.model+' (n='+str(int(len(X0_test)/2))+')')
    plt.plot(tvec, tip_true[:,1], 'k', label='y: '+model_param.model+' (n='+str(int(len(X0_test)/2))+')')
    plt.plot(tvec, tip_true[:,2], 'c', label='z: '+model_param.model+' (n='+str(int(len(X0_test)/2))+')')
if 'MLP' in args.ml_model: 
    tip_MLP = np.array(matlab_eng.tip_position(model_param.beam_data,matlab.double(q_MLP.tolist()), model_param.DoF, nargout=1))
    plt.plot(tvec, tip_MLP[:,0], ':r', label='x: MLP')
    plt.plot(tvec, tip_MLP[:,1], ':g', label='y: MLP')
    plt.plot(tvec, tip_MLP[:,2], ':b', label='z: MLP')
if 'HNN' in args.ml_model: 
    tip_HNN = np.array(matlab_eng.tip_position(model_param.beam_data,matlab.double(q_HNN.tolist()), model_param.DoF, nargout=1))
    plt.plot(tvec, tip_HNN[:,0], '-.r', label='x: HNN')
    plt.plot(tvec, tip_HNN[:,1], '-.g', label='y: HNN')
    plt.plot(tvec, tip_HNN[:,2], '-.b', label='z: HNN')
if 'DHNN' in args.ml_model: 
    tip_DHNN = np.array(matlab_eng.tip_position(model_param.beam_data,matlab.double(q_DHNN.tolist()), model_param.DoF, nargout=1))
    plt.plot(tvec, tip_DHNN[:,0], '--r', label='x: D-HNN')
    plt.plot(tvec, tip_DHNN[:,1], '--g', label='y: D-HNN')
    plt.plot(tvec, tip_DHNN[:,2], '--b', label='z: D-HNN')
plt.xlabel('time [s]')
plt.ylabel('position [m]')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()

#%% Animation:
matlab_eng.animate(model_param.beam_data,"test_NN.mat",args.ml_model,model_param.DoF,n,'gif',nargout=0)

if 'matlab_eng' in locals():
    matlab_eng.quit()
    del matlab_eng
    
#%% Phase portrait for last test run:
# DoF = 0                                                                         # DoF considered for the phase portrait 
# ipython.magic('matplotlib inline')
# fig = plt.figure()
# if args.matlab_comparison:
#     # p_true, q_true = np.split(X_true, 2, axis=1)
#     plt.plot(q_true[:,DoF],p_true[:,DoF], '--k', label='Ground Truth')
# if 'MLP' in args.ml_model: 
#     # p_MLP, q_MLP = np.split(X_MLP, 2, axis=1)
#     plt.plot(q_MLP[:,DoF],p_MLP[:,DoF], 'g', label='MLP')
# if 'HNN' in args.ml_model: 
#     # p_HNN, q_HNN = np.split(X_HNN, 2, axis=1)
#     plt.plot(q_HNN[:,DoF],p_HNN[:,DoF], 'b', label='HNN')
# # if 'DHNN' in args.ml_model: 
#     # p_DHNN, q_DHNN = np.split(X_DHNN, 2, axis=1)
#     # plt.plot(q_DHNN[:,DoF],p_DHNN[:,DoF], 'r', label='D-HNN')
# plt.xlabel('q [rad]')
# plt.ylabel('p [kg*rad/s]')
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.show()
