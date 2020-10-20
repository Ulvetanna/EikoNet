import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from scipy import signal
import torch
import numpy as np
from torch.nn import Linear
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.utils.data.sampler import SubsetRandomSampler,WeightedRandomSampler
from scipy import interpolate
import pandas as pd
from pyproj import Proj
import copy



class _numpy2dataset(torch.utils.data.Dataset):
    def __init__(self, data, target, transform=None):
        # Creating identical pairs
        self.data    = Variable(Tensor(data))
        self.target  = Variable(Tensor(target))

    def send_device(self,device):
        self.data    = self.data.to(device)
        self.target  = self.target.to(device)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y, index
    def __len__(self):
        return self.data.shape[0]

def _randPoints(numsamples=10000,randomDist=False,Xmin=[0,0,0],Xmax=[2,2,2]):
    numsamples = int(numsamples)
    Xmin = np.append(Xmin,Xmin)
    Xmax = np.append(Xmax,Xmax)
    if randomDist:
        X  = np.zeros((numsamples,6))
        PointsOutside = np.arange(numsamples)
        while len(PointsOutside) > 0:
            P  = np.random.rand(len(PointsOutside),3)*(Xmax[:3]-Xmin[:3])[None,None,:] + Xmin[:3][None,None,:]
            dP = np.random.rand(len(PointsOutside),3)-0.5
            rL = (np.random.rand(len(PointsOutside),1))*np.sqrt(np.sum((Xmax-Xmin)**2))
            nP = P + (dP/np.sqrt(np.sum(dP**2,axis=1))[:,np.newaxis])*rL

            X[PointsOutside,:3] = P
            X[PointsOutside,3:] = nP

            maxs          = np.any((X[:,3:] > Xmax[:3][None,:]),axis=1)
            mins          = np.any((X[:,3:] < Xmin[:3][None,:]),axis=1)
            OutOfDomain   = np.any(np.concatenate((maxs[:,None],mins[:,None]),axis=1),axis=1)
            PointsOutside = np.where(OutOfDomain)[0]
    else:
        X  = (np.random.rand(numsamples,6)*(Xmax-Xmin)[None,None,:] + Xmin[None,None,:])[0,:,:]
    return X


def Database(PATH,VelocityFunction,create=False,Numsamples=5000,randomDist=False,SurfaceRecievers=False):
    if create == True:
        xmin = copy.copy(VelocityFunction.xmin)
        xmax = copy.copy(VelocityFunction.xmax)

        # Projecting from LatLong to UTM
        if type(VelocityFunction.projection) == str:
            proj = Proj(VelocityFunction.projection)
            xmin[0],xmin[1] = proj(xmin[0],xmin[1])
            xmax[0],xmax[1] = proj(xmax[0],xmax[1])

        Xp   = _randPoints(numsamples=Numsamples,Xmin=xmin,Xmax=xmax,randomDist=randomDist)
        Yp   = VelocityFunction.eval(Xp)

        while len(np.where(np.isnan(Yp[:,1]))[0]) > 0:
            indx     = np.where(np.isnan(Yp[:,1]))[0]
            print('Recomputing for {} points with nans'.format(len(indx)))
            Xpi      = _randPoints(numsamples=len(indx),Xmin=xmin,Xmax=xmax,randomDist=randomDist)
            Yp[indx,:] = VelocityFunction.eval(Xpi)
            Xp[indx,:] = Xpi

        # Saving the training dataset
        np.save('{}/Xp'.format(PATH),Xp)
        np.save('{}/Yp'.format(PATH),Yp)
    else:
        try:
            Xp = np.load('{}/Xp.npy'.format(PATH))
            Yp = np.load('{}/Yp.npy'.format(PATH))
        except ValueError:
            print('Please specify a correct source path, or create a dataset')


    print(Xp.shape,Yp.shape)
    database = _numpy2dataset(Xp,Yp)

    return database


#==============================================================================================================================
#==============================================================================================================================
#==============================================================================================================================
#==============================================================================================================================

# ---------------- Velocity Functions - Toy Problems ----------------
class ToyProblem_Homogeneous:
    def __init__(self):
        self.xmin       = [0,0,0]
        self.xmax       = [20.,20.,20.]
        self.projection = None
        self.velocity = 5.

    def eval(self,Xp):
        Yp  = np.ones((Xp.shape[0],3))*self.velocity
        return Yp

class ToyProblem_BlockModel:
    def __init__(self):
        self.xmin     = [0,0,0]
        self.xmax     = [20.,20.,20.]

        # projection 
        self.projection = None

        # Velocity values
        self.velocity_outside = 5.
        self.velocity_inside  = 7.

        # 
        self.xmin_inner = [6.,6.,6.]
        self.xmax_inner = [14.,14.,14.]

    def eval(self,Xp):
        Yp  = np.ones((Xp.shape[0],2))*self.velocity_outside
        indS = (Xp[:,0] <= self.xmax_inner[0]) & (Xp[:,0] >= self.xmin_inner[0]) & (Xp[:,1] <= self.xmax_inner[1]) & (Xp[:,1] >= self.xmin_inner[1]) & (Xp[:,2] <= self.xmax_inner[2]) & (Xp[:,2] >= self.xmin_inner[2])
        indR = (Xp[:,3] <= self.xmax_inner[0]) & (Xp[:,3] >= self.xmin_inner[0]) & (Xp[:,4] <= self.xmax_inner[1]) & (Xp[:,4] >= self.xmin_inner[1]) & (Xp[:,5] <= self.xmax_inner[2]) & (Xp[:,5] >= self.xmin_inner[2])
        Yp[indS,0] = self.velocity_inside
        Yp[indR,1] = self.velocity_inside
        return Yp

class ToyProblem_1DGraded:
    def __init__(self):
        self.xmin     = [0,0,0]
        self.xmax     = [20,20,20]

        # projection 
        self.projection = None

        # Velocity values
        self.velocity_min      = 3.0
        self.velocity_gradient = 5.
        self.velcoity_graddim  = 2

    def eval(self,Xp):
        Yp  = np.ones((Xp.shape[0],2))*self.velocity_min
        Yp[:,0] = self.velocity_min + (Xp[:,self.velcoity_graddim])/self.velocity_gradient
        Yp[:,1] = self.velocity_min + (Xp[:,self.velcoity_graddim+3])/self.velocity_gradient
        return Yp

class ToyProblem_Checkerboard:
    def __init__(self):
        self.xmin     = [0,0,0]
        self.xmax     = [20.,20.,20.]

        # projection 
        self.projection = None

        # Velocity values
        self.velocity_mean     = 5.0
        self.velocity_phase    = 0.5
        self.velocity_offset   = -2.5 
        self.velcoity_amp      = 1.0

    def eval(self,Xp):
        Yp = np.ones((Xp.shape[0],2))
        SinS = (signal.square(Xp[:,0]+self.velocity_offset,self.velocity_phase) + signal.square(Xp[:,1]+self.velocity_offset,self.velocity_phase) + signal.square(Xp[:,2]+self.velocity_offset,self.velocity_phase))/3
        SinR = (signal.square(Xp[:,3]+self.velocity_offset,self.velocity_phase) + signal.square(Xp[:,4]+self.velocity_offset,self.velocity_phase) + signal.square(Xp[:,5]+self.velocity_offset,self.velocity_phase))/3
        Yp[:,0] = (SinS)*self.velcoity_amp + self.velocity_mean
        Yp[:,1] = (SinR)*self.velcoity_amp + self.velocity_mean
        return Yp



class Graded1DVelocity:
    def __init__(self,file,xmin=None,xmax=None,projection=None,sep=0.1,polydeg=50):
        self.file        = file
        self.xmin        = xmin
        self.xmax        = xmax

        # projection 
        self.projection  = projection

        self.sep         = sep
        self.velmod      = pd.read_csv(self.file,names=['Depth','V'])
        self.velmod_fnc  = interpolate.interp1d(self.velmod['Depth'],self.velmod['V'])

    def plot(self,file):
        plt.clf()
        X = np.arange(self.xmin[-1],self.xmax[-1],self.sep)
        plt.plot(X,self.velmod_fnc(X),label='Interpolated Velocity')
        plt.scatter(self.velmod['Depth'],self.velmod['V'],15,label='Input velocity')
        plt.ylabel('Velocity km/s')
        plt.xlabel('Depth')
        plt.legend()
        plt.xlim([self.xmin[-1],self.xmax[-1]])
        plt.savefig(file)

    def plot_TestPoints(self,model,file,n=1e5,SurfaceRecievers=False):
        # ----- Plotting Recovered Velocity Misfit ----
        plt.clf()
        number_random_checks = 1000
        
        # Projecting sample points into LatLong
        Xp = torch.rand(int(n),6)
        Xp[:,:3] = Xp[:,:3]*(Tensor(self.xmax)-Tensor(self.xmin))[None,:] + Tensor(self.xmin)
        Xp[:,3:] = Xp[:,3:]*(Tensor(self.xmax)-Tensor(self.xmin))[None,:] + Tensor(self.xmin)

        Vp = model.Velocity(Xp)
        X = np.arange(self.xmin[-1],self.xmax[-1],self.sep)
        plt.scatter(Xp[:,-1].cpu().numpy(),Vp.cpu().detach().numpy(),0.1,'k',label='RecoveredVelocity',alpha=0.1)
        plt.plot(X,self.velmod_fnc(X),'g',label='Interpolated Velocity')
        plt.scatter(self.velmod['Depth'],self.velmod['V'],15,'r',label='Input velocity')
        
        plt.ylabel('Velocity km/s')
        plt.xlabel('Depth')
        plt.legend()
        plt.xlim([self.xmin[-1],self.xmax[-1]])
        plt.savefig(file)

    def eval(self,Xp):
        Yp  = np.zeros((Xp.shape[0],2))
        Yp[:,0] = self.velmod_fnc(Xp[:,2])
        Yp[:,1] = self.velmod_fnc(Xp[:,2+3])
        return Yp








