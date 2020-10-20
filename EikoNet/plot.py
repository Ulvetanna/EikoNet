import matplotlib
matplotlib.use('Agg')
import numpy as np
import math
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
from scipy.ndimage.filters import gaussian_filter
import random
from glob import glob

import torch
from torch.nn import Linear
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.utils.data.sampler import SubsetRandomSampler,WeightedRandomSampler

from mpl_toolkits.mplot3d import Axes3D

import skfmm

import sys

def _plot_TTSlice(fig,model,VelocityFunction,Xsrc,spacing=0.005,device=torch.device('cpu'), TT_FD=None, dims=[0,1], row_plot=1, num_rows=4, vrange=[4,6],contours=True):
    caption_labels = ['X','Y','Z']

    xmin = VelocityFunction.xmin
    xmax = VelocityFunction.xmax
    X,Y      = np.meshgrid(np.arange(xmin[dims[0]],xmax[dims[0]],spacing),np.arange(xmin[dims[1]],xmax[dims[1]],spacing))
    dims_n = np.setdiff1d([0,1,2],dims)[0]

    XP       = np.ones((len(X.flatten()),6))*((xmax[dims_n]-xmin[dims_n])/2 +xmin[dims_n])
    XP[:,:3] = Xsrc
    XP[:,3+dims[0]]  = X.flatten()
    XP[:,3+dims[1]]  = Y.flatten()
    Yp   = VelocityFunction.eval(XP)
    Vobs = Yp[:,1].reshape(X.shape)

    # Determining the travel-time for the points, compute and return to CPU
    XP = Variable(Tensor(XP)).to(device)
    tt = model.TravelTimes(XP)
    vv = model.Velocity(XP)
    TT = tt.to('cpu').data.numpy().reshape(X.shape)
    V  = vv.to('cpu').data.numpy().reshape(X.shape)

    ax    = fig.add_subplot(num_rows,3,(row_plot-1)*3 + 1) 
    quad1 = ax.pcolormesh(X,Y,Vobs,vmin=vrange[0],vmax=vrange[1])


    if TT_FD is not None:
        idx = np.setdiff1d([0,1,2],dims)[0]
        if idx == 0:
            TTobs = TT_FD[int(TT_FD.shape[idx]/2),:,:]
        elif idx ==1:
            TTobs = TT_FD[:,int(TT_FD.shape[idx]/2),:]
        elif idx ==2:
            TTobs = TT_FD[:,:,int(TT_FD.shape[idx]/2)]
        TTobs = np.transpose(TTobs)        
        if contours==True:
            ax.contour(X,Y,TTobs,np.arange(0,10,0.25),colors='w')
    else: 
        TTobs = np.ones(Vobs.shape)*np.nan



    plt.colorbar(quad1,ax=ax, pad=0.1, label='Observed Velocity')
    ax.set_aspect('equal')
    ax.set_title('Imposed Velocity Function')
    ax.set_ylabel('{} location'.format(caption_labels[dims[1]]))
    ax.set_xlabel('{} location'.format(caption_labels[dims[0]]))
    ax.set_xlim([VelocityFunction.xmin[dims[1]],VelocityFunction.xmax[dims[1]]])
    ax.set_ylim([VelocityFunction.xmin[dims[0]],VelocityFunction.xmax[dims[0]]])

    ax = fig.add_subplot(num_rows,3,(row_plot-1)*3 + 2)  
    quad1 = ax.pcolormesh(X,Y,V,vmin=vrange[0],vmax=vrange[1])
    if contours==True:
        ax.contour(X,Y,TT,np.arange(0,10,0.25),colors='w')
    plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')
    ax.set_title('Recovered Velocity Function')
    ax.set_aspect('equal')
    ax.set_ylabel('{} location'.format(caption_labels[dims[1]]))
    ax.set_xlabel('{} location'.format(caption_labels[dims[0]]))
    ax.set_xlim([VelocityFunction.xmin[dims[1]],VelocityFunction.xmax[dims[1]]])
    ax.set_ylim([VelocityFunction.xmin[dims[0]],VelocityFunction.xmax[dims[0]]])

    ax = fig.add_subplot(num_rows,3,(row_plot-1)*3 + 3)  
    quad1 = ax.pcolormesh(X,Y,((V-Vobs)/Vobs)*100,cmap='bwr',vmin=-20.,vmax=20)
    plt.colorbar(quad1,ax=ax, pad=0.1, label='% Diff')
    ax.set_title('Velocity % Diff')
    ax.set_aspect('equal')
    ax.set_ylabel('{} location'.format(caption_labels[dims[1]]))
    ax.set_xlabel('{} location'.format(caption_labels[dims[0]]))
    ax.set_xlim([VelocityFunction.xmin[dims[1]],VelocityFunction.xmax[dims[1]]])
    ax.set_ylim([VelocityFunction.xmin[dims[0]],VelocityFunction.xmax[dims[0]]])
    return fig, (TTobs-TT)#/TTobs

def _plot_LossFunc(fig,model,RMS_TT,num_rows=4,row_plot=5,loss_ylim=[1e-5,1e-2]):
    ax  = fig.add_subplot(num_rows,3, row_plot*3 - 2)
    ax.set_title('Loss Terms - RMS Travel-Time = {}s'.format(RMS_TT))
    quad1 = ax.plot(np.arange(len(model.total_train_loss))+1,model.total_train_loss,'r',label='Training-Loss')
    quad2 = ax.plot(np.arange(len(model.total_train_loss))+1,model.total_val_loss,'k',label='Validation-Loss')
    ax.set_xlim([1,len(np.arange(len(model.total_train_loss))+1)])
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.set_yscale('log')
    ax.set_ylim([loss_ylim[0],loss_ylim[1]])
    #ax.set_xlim([0,200])
    ax.legend(loc=0)
    ax.legend()
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    #plt.suptitle('Training-Loss ={}, Validation-Loss={}\n'.format(model.total_train_loss[-1],model.total_val_loss[-1]))
    return fig


def Plotting(model,Xsrc,spacing=0.1,vrange=[4,6],loss_ylim=[1e-7,1e-2], compute_FD=True,fig_path=None,contours=True):
    VelocityFunction = model.VelocityClass
    plt.clf();plt.close('all')

    if compute_FD==True:
        try:
            print('Computing FD Travel-Time in 3D, this may take some time.')
            xmin      = VelocityFunction.xmin
            xmax      = VelocityFunction.xmax
            X,Y,Z     = np.meshgrid(np.arange(xmin[0],xmax[0],spacing),np.arange(xmin[1],xmax[1],spacing),np.arange(xmin[2],xmax[2],spacing))
            phi       = np.ones(X.shape)
            XP3D      = np.ones((len(X.flatten()),6))
            XP3D[:,0] = X.flatten(); XP3D[:,1] = Y.flatten(); XP3D[:,2] = Z.flatten()
            XP3D[:,3] = X.flatten(); XP3D[:,4] = Y.flatten(); XP3D[:,5] = Z.flatten()
            YP3D = VelocityFunction.eval(XP3D)
            Vobs3D = YP3D[:,1].reshape(X.shape)
            phi[np.argmin(abs(np.arange(xmin[0],xmax[0],spacing)-Xsrc[0])),
                np.argmin(abs(np.arange(xmin[1],xmax[1],spacing)-Xsrc[1])),
                np.argmin(abs(np.arange(xmin[2],xmax[2],spacing)-Xsrc[2]))] = -1
            TT_FD = skfmm.travel_time(phi,Vobs3D,dx=spacing)
        except:
            print('Compute Finite-Difference failed with quadratic fitting error.\n')
            print('This can be experienced in sharp velocitu contrasts.')
            print('Continuing without Finite-Difference Travel-Time !')
            TT_FD = None
            compute_FD=False

    else:
        TT_FD = None


    fig = plt.figure(figsize=(20,10))
    fig, T1 = _plot_TTSlice(fig,model,VelocityFunction,Xsrc,
                     spacing=spacing, device=torch.device('cpu'), TT_FD=TT_FD,
                     dims=[0,1], row_plot=1, num_rows=4,vrange=vrange,contours=contours)
    fig, T2 = _plot_TTSlice(fig,model,VelocityFunction,Xsrc,
                     spacing=spacing, device=torch.device('cpu'), TT_FD=TT_FD,
                     dims=[0,2], row_plot=2, num_rows=4,vrange=vrange,contours=contours)
    fig, T3 = _plot_TTSlice(fig,model,VelocityFunction,Xsrc,
                     spacing=spacing, device=torch.device('cpu'), TT_FD=TT_FD,
                     dims=[1,2], row_plot=3, num_rows=4,vrange=vrange,contours=contours)
    RMS_TT     = np.sqrt(np.mean(T1**2 + T2**2 + T3**2))#*100
    fig = _plot_LossFunc(fig,model,RMS_TT,row_plot=4, num_rows=4,loss_ylim=loss_ylim)



    if fig_path != None:
        plt.savefig(fig_path,dpi=600)

    return fig
