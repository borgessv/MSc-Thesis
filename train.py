# Dissipative Hamiltonian Neural Networks
# Andrew Sosanya, Sam Greydanus
import torch
import matplotlib.pyplot as plt
import numpy as np
import time

# plots are shown in separate window:
from IPython import get_ipython
ipython = get_ipython()

def get_batch(v, step, args):  # helper function for moving batches of data to/from GPU
  dataset_size, num_features = v.shape
  bix = (step*args.batch_size) % dataset_size
  v_batch = v[bix:bix + args.batch_size, :]  # select next batch
  return torch.tensor(v_batch, requires_grad=True,  dtype=torch.float32, device=args.device)


def train(model, args, data):
  ipython.magic('matplotlib auto')
  """ General training function"""
  model = model.to(args.device)  # put the model on the GPU (if enabled)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay)  # setting the Optimizer

  model.train()     # doesn't make a difference for now
  t0 = time.time()  # logging the time
  results = {'train_loss':[], 'test_loss':[], 'test_acc':[], 'global_step':0}  # Logging the results
  
  x_step, y1_step, y2_step = [], [], []
  plt.close()
  plt.ion()       # Enable interactive mode
  fig, ax = plt.subplots()
  ax.set_autoscale_on(True) # enable autoscale
  ax.autoscale_view(True,True,True)
  line1, line2, = ax.plot([], [], 'b-', [], [], 'r-') # Plot blank data
  plt.yscale('log')
  plt.xlabel('epoch')
  plt.ylabel('L2 error')
  plt.legend(['train','test'],loc='upper right')
  plt.grid(True, which='both')
  
  for step in range(args.total_steps):  # training loop 

    x, t, dx = [get_batch(data[k], step, args) for k in ['x', 't', 'dx']]
    optimizer.zero_grad()
    dx_hat = model(x, t=t)  # feeding forward
    loss = (dx-dx_hat).pow(2).mean()  # L2 loss function
    loss.backward(); optimizer.step()  # backpropogation

    results['train_loss'].append(loss.item())  # logging the training loss

    # Testing our data with desired frequency (test_every)
    if step % args.test_every == 0:

      x_test, t_test, dx_test = [get_batch(data[k], step, args=args) for k in ['x_test', 't_test', 'dx_test']]
      dx_hat_test = model(x_test, t=t_test)  # testing our model. 
      test_loss = (dx_test-dx_hat_test).pow(2).mean().item() # L2 loss

      results['test_loss'].append(test_loss)
    if step % args.print_every == 0:
      print('step {}, dt {:.3f}, train_loss {:.2e}, test_loss {:.2e}'
            .format(step, time.time()-t0, loss.item(), test_loss)) #.item() is just the integer of PyTorch scalar. 
      t0 = time.time()
      
    if step % 50 == 0:  
        # if 'ax' in globals(): ax.remove()
        x_step.append(step)
        y1_step.append(loss.item())
        y2_step.append(test_loss)
        line1.set_data(x_step, y1_step)
        line2.set_data(x_step, y2_step)
        ax.relim()        # Recalculate limits
        ax.autoscale_view(True,True,True) #Autoscale
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.0001)
 
  model = model.cpu()
  return results