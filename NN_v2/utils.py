# -*- coding: utf-8 -*-
"""
"""
import keras
import torch
import matplotlib.pyplot as plt
import numpy as np
import os, sys, io, time
import imageio
import matlab.engine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
from IPython.display import clear_output
from PIL import Image
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
    

def pre_process(data_raw, model_param):
    n_train = round(model_param.n_samples*model_param.train_split)                  # train/test split:
    split_index = int(n_train*len(data_raw['x'])/model_param.n_samples)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k + '_train'], split_data[k + '_val'] = data_raw[k][:split_index], data_raw[k][split_index:]
    data_raw = split_data
        
    data = {}
    if model_param.normalize:                                                       # normalizes/standardizes dataset (if enabled)
        x_scaler = MinMaxScaler(feature_range=model_param.norm_range)
        dx_scaler = MinMaxScaler(feature_range=model_param.norm_range)
        data['x_train'] = x_scaler.fit_transform(data_raw['x_train'])
        data['dx_train'] = dx_scaler.fit_transform(data_raw['dx_train'])
        data['x_val'] = x_scaler.transform(data_raw['x_val'])
        data['dx_val'] = dx_scaler.transform(data_raw['dx_val'])
    elif model_param.standardize:
        x_scaler = StandardScaler()
        dx_scaler = StandardScaler()
        data['x_train'] = x_scaler.fit_transform(data_raw['x_train'])
        data['dx_train'] = dx_scaler.fit_transform(data_raw['dx_train'])
        data['x_val'] = x_scaler.transform(data_raw['x_val'])
        data['dx_val'] = dx_scaler.transform(data_raw['dx_val'])
    else:
        for k in ['x_train', 'dx_train', 'x_val', 'dx_val']:
            data[k] = data_raw[k]
            x_scaler, dx_scaler = [], []
    return data, x_scaler, dx_scaler


def meta_model(t, x, model, x_scaler, dx_scaler, data_args):
    # t = time.time()
    # print(x.reshape(-1,1))
    if data_args.external_force:
        Qe = data_args.matlab_engine.external_forces(matlab.double(t),matlab.double(x.reshape(-1,1)),data_args.beam_data,data_args.DoF,
                                                data_args.gravity,data_args.model,data_args.phi_r,nargout=1)
    # print(t)
    x = x_scaler.transform(x.reshape(1,-1)) if x_scaler != [] else x.reshape(1,-1)
    x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    # t1 = time.time() - t
    # t = time.time()
    dx = model(x)['x_hat']

    # t2 = time.time() - t
    # t = time.time()
    dx = dx_scaler.inverse_transform(dx.data.numpy()).flatten() if dx_scaler != [] else dx.data.numpy().flatten()
    if data_args.external_force:
        dx[:len(dx)//2] += np.array(Qe).flatten()
        # print(np.array(Qe).flatten())
    # print(dx[:len(dx)//2].shape)
    # t3 = time.time() - t
    # print([t1,t2,t3])
    return dx 


def post_process(X, model_param, phi_r):                         
    if model_param.model == 'FOM':
        p, q = np.split(X, 2, axis=1)  
    elif model_param.model == 'ROM':   
        etap, eta = np.split(X, 2, axis=1)
        p, q = (phi_r@etap.T).T, (phi_r@eta.T).T
    return p, q
        
            
class PlotCallback(keras.callbacks.Callback):                                   # Callback function to store losses at each epoch and create gif of the train evolution including a test example
    def __init__(self, epochs):
        # self.test = test_img
        # self.model = model
        self.epochs = epochs
        
        
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
        # self.train_img_folder = 'train_img_'+datetime.now().strftime('%d-%m-%Y-%H:%M:%S')
        # self.img2gif_folder = 'img2gif_'+datetime.now().strftime('%d-%m-%Y-%H:%M:%S')
        # os.mkdir(self.train_img_folder)
        # os.mkdir(self.img2gif_folder)
        self.frames = []
            
    def on_epoch_end(self, epoch, logs={}):
        # pred_img = self.model.predict(np.expand_dims(self.test,axis=0), verbose=0)
        # pred_img = pred_img.squeeze()
        # pred_img = pred_img*255
        # cv2.imwrite(os.path.join(self.train_img_folder, str(epoch)+'.png'), pred_img.astype(np.uint8))
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        metrics = ['loss']
        fig, ax = plt.subplots()
        clear_output(wait=True)
        for i, metric in enumerate(metrics):
            ax.plot(range(1, epoch + 2), self.metrics[metric], 'b', label='train')
            if logs['val_' + metric]:
                ax.plot(range(1, epoch + 2), self.metrics['val_' + metric], 'r', label='validation')
            ax.set_yscale('log')    
            ax.legend(loc='upper right')
            ax.grid(which='both')
        plt.title('epoch: %i' %(epoch+1), loc='right', weight='bold')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.xlim([1, self.epochs])
        plt.tight_layout()
        # attach_img = plt.imread(os.path.join(self.train_img_folder, str(epoch)+'.png'))
        # ax2 = fig.add_axes([0.55,0.5,0.4,0.4], anchor='NE', zorder=1)
        # ax2.imshow(attach_img)
        # ax2.axis('off')
        image_buffer = io.BytesIO()
        plt.savefig(image_buffer, format='png', dpi=300, bbox_inches='tight') 
        image_buffer.seek(0)
        new_frame = imageio.imread(image_buffer)
        self.frames.append(new_frame)
        plt.close()

        
    def on_train_end(self, logs={}):
        # self.frames[0].save('train_evolution_'+datetime.now().strftime('%d-%m-%Y-%H:%M:%S')+'gif', 
        #                     format='GIF', append_images=self.frames[1:], save_all=True, 
        #                     duration=100, loop=0)
        imageio.mimsave('train_evolution_' + datetime.now().strftime('%d-%m-%Y-%H:%M:%S') + '.gif', 
                        self.frames, format='GIF', duration=0.1, loop=0)
        
        metrics = ['loss']
        fig, ax = plt.subplots()
        clear_output(wait=True)
        for i, metric in enumerate(metrics):
            ax.plot(range(1, self.epochs + 1), self.metrics[metric], 'b', label='train')
            if logs['val_' + metric]:
                ax.plot(range(1, self.epochs + 1), self.metrics['val_' + metric], 'r', label='validation')
            ax.set_yscale('log')    
            ax.legend(loc='upper right')
            ax.grid(which='both')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.xlim([1, self.epochs])
        plt.tight_layout()
        plt.show()
        plt.savefig('loss_plot_'+datetime.now().strftime('%d-%m-%Y-%H:%M:%S')+'.png',
                    dpi=300, bbox_inches='tight')
                    




    