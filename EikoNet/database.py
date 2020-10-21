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



class SCEC_CVMH:
    def __init__(self,xmin=None,xmax=None,projection=None,phase='VP',cvm_host=None):
        self.xmin         = xmin
        self.xmax         = xmax
        self.projection   = projection

        self.FileTemp = os.getcwd()
  
    if type(self.cvm_host) == type(None):
      print('Please specify a path to the CVM-H Fortran code')

        self.phase    = phase

    def eval(self,Xp):
        Yp = np.zeros((Xp.shape[0],2))

        print('Compute CVM-H at point locations {}'.format(len(Yp)))

        # converting back into LatLong and flattening
        proj               = Proj(self.projection)
        long_flat,lat_flat = proj(Xp[:,3],Xp[:,4],inverse=True) 

        # Creating a grid of points and saving to a temp file
        Locations = pd.DataFrame({'X':long_flat,'Y':lat_flat,'Z':-Xp[:,5]*1000})
        Locations.to_csv('{}/tmp_events'.format(self.FileTemp),header=False,index=False,sep=' ')

        # Running CVM-H 
        call('../src/vx < {}/tmp_events > {}/tmp_vpvs'.format(self.FileTemp,self.FileTemp),cwd='{}/model'.format(self.cvm_host),shell=True)
        VPVS = pd.read_csv('tmp_vpvs',
                            names=['Long','Lat','Z','UTM_X','UTM_Y','UTM_elv_X','UTM_elv_Y',
                                   'topo','mtop','base','moho','flg','cellX','cellY','cellZ',
                                   'tag','VP','VS','RHO'],sep=r'\s+')

        if self.phase == 'VP':
            Yp[:,1] = VPVS['VP']
        if self.phase == 'VS':
            Yp[:,1] = VPVS['VS']
        
        # Removing unknown velocities
        Yp[Yp[:,1]==-99999.0000] = np.nan
        Yp[Yp[:,1]==0.0] = np.nan

        # Velocity in km/s
        Yp = Yp/1000

        return Yp


    def Plotting(self,TT_model,Xp,Yp,phase='VP',Xsrc=None,indexVal=None,save_path=None):
        '''
        '''
        if type(Xsrc) == type(None):
            Xsrc = (np.array(self.xmax) - np.array(self.xmin))/2 + np.array(self.xmin)
        if type(indexVal) == type(None):
            indexVal = [0,((np.array(self.xmax) - np.array(self.xmin))/2 + np.array(self.xmin))[0],0.03]


        Xp = np.load(Xp)
        Yp = np.load(Yp)

        # Projecting sample points into LatLong
        proj = Proj(self.projection)
        Xp[:,0],Xp[:,1] = proj(Xp[:,0],Xp[:,1],inverse=True)
        Xp[:,3],Xp[:,4] = proj(Xp[:,3],Xp[:,4],inverse=True)



        dms = [0,1,2]
        dms.remove(indexVal[0])

        # === Gridding the correct data
        if indexVal[0] == 0:
            indx      = np.where(abs(Xp[:,3] - indexVal[1])<indexVal[2])[0]
        if indexVal[0] == 1:
            indx      = np.where(abs(Xp[:,4] - indexVal[1])<indexVal[2])[0]
        if indexVal[0] == 2:
            indx      = np.where(abs(Xp[:,5] - indexVal[1])<indexVal[2])[0]

        # Determining the Travel-time and velocity at points
        Xp_i = Xp[indx,:]
        Points       = np.zeros((len(indx),6))
        Points[:,:3] = Xsrc
        Points[:,3]  = Xp_i[:,3]
        Points[:,4]  = Xp_i[:,4]
        Points[:,5]  = Xp_i[:,5]
        Points       = torch.Tensor(Points)

        ObsVV        = Yp[indx,1]

        TT = TT_model.TravelTimes(Points).detach().cpu().numpy()
        VV = TT_model.Velocity(Points).detach().cpu().numpy()

        plt.clf();plt.close('all')
        fig = plt.figure(figsize=(20,8))
        dim_labs = ['Long','Lat','Z']
        plt.suptitle('Xsrc = [{},{},{}],\n Slice in {} direction taken at {}={} +/- {}'.format(Xsrc[0],Xsrc[1],Xsrc[2],dim_labs[indexVal[0]],dim_labs[indexVal[0]],indexVal[1],indexVal[2]))

        ax0  = fig.add_subplot(2,2,1)
        ax0.set_title('Observed Velocity (km/s)')
        quad0 = ax0.scatter(Xp_i[:,dms[0]+3],Xp_i[:,dms[1]+3],5,ObsVV)
        cb0=plt.colorbar(quad0,ax=ax0);cb0.set_label('CVM-H Vel (km/s)')
        ax0.set_xlabel('{}'.format(dim_labs[dms[0]]))
        ax0.set_ylabel('{}'.format(dim_labs[dms[1]]))
        ax0.set_xlim([self.xmin[dms[0]],self.xmax[dms[0]]])
        ax0.set_ylim([self.xmax[dms[1]],self.xmin[dms[1]]])

        ax  = fig.add_subplot(2,2,2)
        ax.set_title('Predicted Travel-Time (s)')
        quad = ax.scatter(Xp_i[:,dms[0]+3],Xp_i[:,dms[1]+3],5,TT,cmap='hsv')
        cb = plt.colorbar(quad,ax=ax);cb.set_label('Predicted TT (s)')
        ax.set_xlabel('{}'.format(dim_labs[dms[0]]))
        ax.set_ylabel('{}'.format(dim_labs[dms[1]]))
        ax.set_xlim([self.xmin[dms[0]],self.xmax[dms[0]]])
        ax.set_ylim([self.xmax[dms[1]],self.xmin[dms[1]]])

        ax1  = fig.add_subplot(2,2,3)
        ax1.set_title('Predicted Velocity Model (km/s)')
        quad1 = ax1.scatter(Xp_i[:,dms[0]+3],Xp_i[:,dms[1]+3],5,VV)
        cb1=plt.colorbar(quad1,ax=ax1);cb1.set_label('Predicted Vel (km/s)')
        if (indexVal[0] == 0) or (indexVal[0] ==1):
            ax1.invert_yaxis()
        ax1.set_xlabel('{}'.format(dim_labs[dms[0]]))
        ax1.set_ylabel('{}'.format(dim_labs[dms[1]]))
        ax1.set_xlim([self.xmin[dms[0]],self.xmax[dms[0]]])
        ax1.set_ylim([self.xmax[dms[1]],self.xmin[dms[1]]])

        ax2  = fig.add_subplot(2,2,4)
        ax2.set_title('Velocity Model % Difference')
        quad2 = ax2.scatter(Xp_i[:,dms[0]+3],Xp_i[:,dms[1]+3],5,((VV-ObsVV)/ObsVV)*100,cmap='bwr',vmin=-25,vmax=25)
        if (indexVal[0] == 0) or (indexVal[0] ==1):
            ax2.invert_yaxis()
        cb2=plt.colorbar(quad2,ax=ax2);cb2.set_label('Percentage Diff (%)')
        ax2.set_xlabel('{}'.format(dim_labs[dms[0]]))
        ax2.set_ylabel('{}'.format(dim_labs[dms[1]]))
        ax2.set_xlim([self.xmin[dms[0]],self.xmax[dms[0]]])
        ax2.set_ylim([self.xmax[dms[1]],self.xmin[dms[1]]])

        # Input comparison of points 
        if type(save_path) == str:
            plt.savefig(save_path)
        else:
            plt.show()







