import matplotlib
import numpy as np
import math
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
import random
from glob import glob
import time

import torch
from torch.nn import Linear
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast
from torch.utils.data.sampler import SubsetRandomSampler,WeightedRandomSampler
from EikoNet import database as db
from pyproj import Proj
import copy


def init_weights(m):
    if type(m) == torch.nn.Linear:
        stdv = (1. / math.sqrt(m.weight.size(1))/1.)*2
        m.weight.data.uniform_(-stdv,stdv)
        m.bias.data.uniform_(-stdv,stdv)

class NN(torch.nn.Module):
    def __init__(self, nl=10, activation=torch.nn.ELU()):
            super(NN, self).__init__()
            self.act = activation

            # Input Structure
            self.fc0  = Linear(2*3,32)
            self.fc1  = Linear(32,512)

            # Resnet Block
            self.rn_fc1 = torch.nn.ModuleList([Linear(512, 512) for i in range(nl)])
            self.rn_fc2 = torch.nn.ModuleList([Linear(512, 512) for i in range(nl)])
            self.rn_fc3 = torch.nn.ModuleList([Linear(512, 512) for i in range(nl)])

            # Output structure
            self.fc8  = Linear(512,32)
            self.fc9  = Linear(32,1)

    def forward(self,x):
        x   = self.act(self.fc0(x))
        x   = self.act(self.fc1(x))
        for ii in range(len(self.rn_fc1)):
            x0 = x
            x  = self.act(self.rn_fc1[ii](x))
            x  = self.act(self.rn_fc3[ii](x)+self.rn_fc2[ii](x0))

        x     = self.act(self.fc8(x))
        tau   = abs(self.fc9(x))
        return tau

def EikonalLoss(Yobs,Xp,tau,device):
        dtau  = torch.autograd.grad(outputs=tau, inputs=Xp, grad_outputs=torch.ones(tau.size()).to(device), 
                            only_inputs=True,create_graph=True,retain_graph=True)[0]

        T0    = torch.sqrt(((Xp[:,3]-Xp[:,0])**2 + (Xp[:,4]-Xp[:,1])**2 + (Xp[:,5]-Xp[:,2])**2))  
        T1    = (T0**2)*(dtau[:,3]**2 + dtau[:,4]**2 + dtau[:,5]**2)
        T2    = 2*tau[:,0]*(dtau[:,3]*(Xp[:,3]-Xp[:,0]) + dtau[:,4]*(Xp[:,4]-Xp[:,1]) + dtau[:,5]*(Xp[:,5]-Xp[:,2]))
        T3    = tau[:,0]**2
        S2    = (T1+T2+T3)
        Ypred = torch.sqrt(1/S2)
        diff  = abs(Yobs[:,1]-Ypred)/Yobs[:,1]
        loss  = torch.mean(abs((Yobs[:,1]-Ypred)/Yobs[:,1]))
        return loss, diff


class Model():
    def __init__(self, ModelPath, VelocityClass, device='cpu'):
        
        # ======================= JSON Template =======================
        self.Params = {}
        self.Params['ModelPath']     = ModelPath
        self.Params['VelocityClass'] = VelocityClass #Pass the JSON information
        self.Params['Device']        = device
        self.Params['Pytorch Amp (bool)'] = False

        self.Params['Network']  = {}
        self.Params['Network']['Number of Residual Blocks']  = 10
        self.Params['Network']['Layer activation']           = torch.nn.ELU()
        self.Params['Network']['Normlisation']               = 'OffsetMinMax'

        self.Params['Training'] = {}
        self.Params['Training']['Number of sample points']   = 1e6
        self.Params['Training']['Batch Size']                = 752
        self.Params['Training']['Validation Percentage']     = 10
        self.Params['Training']['Number of Epochs']          = 200
        self.Params['Training']['Resampling Bounds']         = [0.1,0.9]
        self.Params['Training']['Print Every * Epoch']       = 1
        self.Params['Training']['Save Every * Epoch']        = 10
        self.Params['Training']['Learning Rate']             = 5e-5
        self.Params['Training']['Random Distance Sampling']  = True
        self.Params['Training']['Use Scheduler (bool)']      = True

        # Parameters to alter during training
        self.total_train_loss = []
        self.total_val_loss   = []


    def _init_network(self):
        self.network = NN(nl=self.Params['Network']['Number of Residual Blocks'],activation=self.Params['Network']['Layer activation'])
        self.network.apply(init_weights)
        self.network.float()
        self.network.to(torch.device(self.Params['Device']))



    def _projection(self,Xp,inverse=False):
        if type(self.Params['VelocityClass'].projection) != type(None):
            proj = Proj(self.Params['VelocityClass'].projection)
            Xp = Xp.detach().cpu().numpy()
            Xp[:,0],Xp[:,1] = proj(Xp[:,0],Xp[:,1],inverse=inverse)
            Xp[:,3],Xp[:,4] = proj(Xp[:,3],Xp[:,4],inverse=inverse)
            Xp = torch.Tensor(Xp)
            Xp = Xp.to(torch.device(self.Params['Device']))
        return Xp


    def  _normalization(self,Xp=None,Yp=None):

        # Loading the predefined variables
        if self.Params['Network']['Normlisation'] == 'MinMax':
            xmin_UTM = np.array(copy.copy(self.Params['VelocityClass'].xmin))
            xmax_UTM = np.array(copy.copy(self.Params['VelocityClass'].xmax))
            if type(self.Params['VelocityClass'].projection) == str:
                proj = Proj(self.Params['VelocityClass'].projection)
                xmin_UTM[0],xmin_UTM[1] = proj(xmin_UTM[0],xmin_UTM[1]) 
                xmax_UTM[0],xmax_UTM[1] = proj(xmax_UTM[0],xmax_UTM[1]) 
            indx = np.argmax(xmax_UTM-xmin_UTM)
            self.nf_max    = xmax_UTM[indx]
            self.nf_min    = xmin_UTM[indx]
            self.sf        = (self.nf_max-self.nf_min)

            if (type(Xp)!=type(None)) and (type(Yp)==type(None)):
                Xp  = Xp/self.sf
                return Xp
            if (type(Xp)==type(None)) and (type(Yp)!=type(None)):
                Yp  = Yp*self.sf
                return Yp
            else:
                Xp = Xp/self.sf
                Yp = Yp/self.sf
                return Xp,Yp

        if self.Params['Network']['Normlisation'] == 'OffsetMinMax':
            xmin_UTM = np.array(copy.copy(self.Params['VelocityClass'].xmin))
            xmax_UTM = np.array(copy.copy(self.Params['VelocityClass'].xmax))
            if type(self.Params['VelocityClass'].projection) == str:
                proj = Proj(self.Params['VelocityClass'].projection)
                xmin_UTM[0],xmin_UTM[1] = proj(xmin_UTM[0],xmin_UTM[1]) 
                xmax_UTM[0],xmax_UTM[1] = proj(xmax_UTM[0],xmax_UTM[1]) 
            indx = np.argmax(xmax_UTM-xmin_UTM)
            self.nf_max    = xmax_UTM[indx]
            self.nf_min    = xmin_UTM[indx]
            self.sf        = (self.nf_max-self.nf_min)

            self.crt_point = (xmax_UTM - xmin_UTM)/2 + xmin_UTM

            if (type(Xp)!=type(None)) and (type(Yp)==type(None)):
                for ii in [0,1,2]:
                    Xp[:,ii]   = Xp[:,ii]   - self.crt_point[ii]
                    Xp[:,ii+3] = Xp[:,ii+3] - self.crt_point[ii]
                Xp = (Xp)/self.sf
                return Xp
            if (type(Xp)==type(None)) and (type(Yp)!=type(None)):
                Yp  = Yp*self.sf
                return Yp
            else:
                for ii in [0,1,2]:
                    Xp[:,ii]   = Xp[:,ii]   - self.crt_point[ii]
                    Xp[:,ii+3] = Xp[:,ii+3] - self.crt_point[ii]
                Xp = (Xp)/self.sf
                Yp = (Yp)/self.sf
                return Xp,Yp

    def train(self):


        # Initialising the network
        self._init_network()


        # Defining the optimization scheme
        self.optimizer  = torch.optim.Adam(self.network.parameters(),lr=self.Params['Training']['Learning Rate'])
        if self.Params['Training']['Use Scheduler (bool)'] == True:
            self.scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)



        # Creating a sampling dataset
        self.dataset        = db.Database(self.Params['ModelPath'],self.Params['VelocityClass'],create=True,Numsamples=int(self.Params['Training']['Number of sample points']),randomDist=self.Params['Training']['Random Distance Sampling'])
        self.dataset.send_device(torch.device(self.Params['Device']))
        self.dataset.data,self.dataset.target = self._normalization(Xp=self.dataset.data,Yp=self.dataset.target)


        # 
        len_dataset         = len(self.dataset)
        n_batches           = int(len(self.dataset)/int(self.Params['Training']['Batch Size']) + 1)
        training_start_time = time.time()

        # --------- Splitting the dataset into training and validation -------
        indices            = list(range(int(len_dataset)))
        validation_idx     = np.random.choice(indices, size=int(len_dataset*(self.Params['Training']['Validation Percentage']/100)), replace=False)
        train_idx          = list(set(indices) - set(validation_idx))
        validation_sampler = SubsetRandomSampler(validation_idx)
        train_sampler      = SubsetRandomSampler(train_idx)

        train_loader       = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=int(self.Params['Training']['Batch Size'] ),
            sampler=train_sampler,
            )    
        validation_loader  = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=int(self.Params['Training']['Batch Size'] ),
            sampler=validation_sampler,
        )    



        # Defining the initial weights to sample by
        weights = Tensor(torch.ones(len(self.dataset))).to(torch.device(self.Params['Device']))
        weights[validation_idx] = 0.0
        print(weights.device)

        print("=======================================================================================")
        print("=======================================================================================")
        print("========================== Eikonal Solver - Training ==================================")
        print("=======================================================================================")
        print("=======================================================================================")
        # Add in the dumping of the JSON file

        print("=======================================================================================")

        for epoch in range(1,self.Params['Training']['Number of Epochs']+1):
            print_every           = 1
            start_time            = time.time()
            running_sample_count  = 0
            total_train_loss      = 0
            total_val_loss        = 0

            # --- Defining the weighting of the samples
            weights                 = torch.clamp(weights/weights.max(),self.Params['Training']['Resampling Bounds'][0],self.Params['Training']['Resampling Bounds'][1])
            weights[validation_idx] = 0.0
            train_sampler_wei       = WeightedRandomSampler(weights, len(weights), replacement=True)
            train_loader_wei        = torch.utils.data.DataLoader(
                                        self.dataset,
                                        batch_size=int(self.Params['Training']['Batch Size'] ),
                                        sampler=train_sampler_wei,
                                      )
            weights                 = Tensor(torch.zeros(len(self.dataset))).to(torch.device(self.Params['Device']))

            for i, data in enumerate(train_loader_wei, 0):
                #print('----------------- Epoch {} - Batch {} --------------------'.format(epoch,i))
                
                # Get inputs/outputs and wrap in variable object
                inputs, labels, indexbatch = data
                inputs = inputs.float()
                labels = labels.float()

                inputs.requires_grad_()


                if self.Params['Pytorch Amp (bool)']:
                    with autocast():
                        outputs = self.network(inputs)
                        loss_value, wv  = EikonalLoss(labels,inputs,outputs,torch.device(self.Params['Device']))
                else:
                    outputs = self.network(inputs)
                    loss_value, wv  = EikonalLoss(labels,inputs,outputs,torch.device(self.Params['Device']))


                loss_value.backward()

                # Update parameters
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Updating the weights
                weights[indexbatch] = wv

                total_train_loss += loss_value.item()
                del inputs, labels, indexbatch, outputs, loss_value, wv


            # ----- Determining the Training Loss -----
            for i, data_val in enumerate(validation_loader, 0):
                inputs_val, labels_val, indexbatch_val = data_val
                inputs_val = inputs_val.float()
                labels_val = labels_val.float()
                inputs_val.requires_grad_()

                if self.Params['Pytorch Amp (bool)']:
                    with autocast():
                        outputs_val                 = self.network(inputs_val)
                        val_loss,wv                 = EikonalLoss(labels_val,inputs_val,outputs_val,torch.device(self.Params['Device']))
                else:
                    outputs_val  = self.network(inputs_val)
                    val_loss,wv  = EikonalLoss(labels_val,inputs_val,outputs_val,torch.device(self.Params['Device']))

                total_val_loss             += val_loss.item()
                del inputs_val, labels_val, indexbatch_val, outputs_val, val_loss, wv


            # Creating a running loss for both training and validation data
            total_val_loss   /= len(validation_loader)
            total_train_loss /= len(train_loader)
            self.total_train_loss.append(total_train_loss)
            self.total_val_loss.append(total_val_loss)

            if self.Params['Training']['Use Scheduler (bool)'] == True:
                self.scheduler.step(total_val_loss)

            del train_loader_wei,train_sampler_wei

            if epoch % self.Params['Training']['Print Every * Epoch'] == 0:
                with torch.no_grad():
                    print("Epoch = {} -- Training loss = {:.4e} -- Validation loss = {:.4e}".format(epoch,total_train_loss,total_val_loss))

            if (epoch % self.Params['Training']['Save Every * Epoch'] == 0) or (epoch == self.Params['Training']['Number of Epochs'] ) or (epoch == 1):
                with torch.no_grad():
                    self.save(epoch=epoch,val_loss=total_val_loss)

    def save(self,epoch='',val_loss=''):
        '''
            Saving a instance of the model
        '''
        torch.save({'epoch':epoch,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': self.total_train_loss,
                    'val_loss': self.total_val_loss}, '{}/Model_Epoch_{}_ValLoss_{}.pt'.format(self.Params['ModelPath'],str(epoch).zfill(5),val_loss))

    def load(self,filepath):
        # -- Loading model information
        self._init_network()
        checkpoint            = torch.load(filepath,map_location=torch.device(self.Params['Device']))
        self.total_train_loss = checkpoint['train_loss']
        self.total_val_loss   = checkpoint['val_loss']
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.to(torch.device(self.Params['Device']))


    def TravelTimes(self,Xp,projection=True,normlisation=True):
        # Apply projection from LatLong to UTM
        Xp  = Xp.to(torch.device(self.Params['Device']))
        if projection:
            Xp  = self._projection(Xp)
        if normlisation:
            Xp  = self._normalization(Xp=Xp,Yp=None)
        tau = self.network(Xp)
        T0  = torch.sqrt(((Xp[:,3]-Xp[:,0])**2 + (Xp[:,4]-Xp[:,1])**2 + (Xp[:,5]-Xp[:,2])**2))
        TT  = tau[:,0]*T0
        del Xp,tau,T0
        return TT

    def Velocity(self,Xp,projection=True,normlisation=True):
        Xp    = Xp.to(torch.device(self.Params['Device']))
        if projection:
            Xp  = self._projection(Xp)
        if normlisation:
            Xp  = self._normalization(Xp=Xp,Yp=None)
        Xp.requires_grad_()
        tau   = self.network(Xp)
        dtau  = torch.autograd.grad(outputs=tau, inputs=Xp, grad_outputs=torch.ones(tau.size()).to(torch.device(self.Params['Device'])), 
                                    only_inputs=True,create_graph=True,retain_graph=True)[0]        
        T0    = torch.sqrt(((Xp[:,3]-Xp[:,0])**2 + (Xp[:,4]-Xp[:,1])**2 + (Xp[:,5]-Xp[:,2])**2))  
        T1    = (T0**2)*(dtau[:,3]**2 + dtau[:,4]**2 + dtau[:,5]**2)
        T2    = 2*tau[:,0]*(dtau[:,3]*(Xp[:,3]-Xp[:,0]) + dtau[:,4]*(Xp[:,4]-Xp[:,1]) + dtau[:,5]*(Xp[:,5]-Xp[:,2]))
        T3    = tau[:,0]**2
        Ypred = torch.sqrt(1/(T1+T2+T3))
        Ypred = self._normalization(Yp=Ypred)
        del Xp,tau,dtau,T0,T1,T2,T3
        return Ypred

    def StationayPoints(self,Xsrc,Xrcv,Xpoints=None,numPoints=1e4):
        ''' 
            Supplying a series of Source-Reciever locations. 

            This function returrns a the traveltime between the two and the stationary value.


            #TO DO:
                ADD LAT/LOT conversion of input points


        '''


        Xsrc = np.array(Xsrc)
        Xrcv = np.array(Xrcv)


        if Xpoints == None:
            XPs = np.array(np.random.rand(int(numPoints),3)*(self.Params['VelocityClass'].xmax-self.Params['VelocityClass'].xmin)[None,:] + self.Params['VelocityClass'].xmin[None,:])
        else:
            XPs = np.array(Xpoints)

        if type(self.Params['VelocityClass'].projection) != type(None):
            proj = Proj(self.Params['VelocityClass'].projection)
            Xsrc[0],Xsrc[1] = proj(Xsrc[0],Xsrc[1])
            Xrcv[0],Xrcv[1]     = proj(Xrcv[0],Xrcv[1])
            XPs[:,0],XPs[:,1]   = proj(XPs[:,0],XPs[:,1])


        XPs  = Variable(Tensor(XPs)).to(torch.device(self.Params['Device']))
        XSs  = Variable(torch.ones((XPs.shape[0],3))*(Xsrc[None,:])).to(torch.device(self.Params['Device']))
        XRs  = Variable(torch.ones((XPs.shape[0],3))*(Xrcv[None,:])).to(torch.device(self.Params['Device']))
        XPs  = XPs.float(); XSs = XSs.float(); XRs = XRs.float(); 
        XPs.requires_grad_()

        T_s2p = self.TravelTimes(torch.cat((XSs,XPs),dim=1),projection=False)
        T_r2p = self.TravelTimes(torch.cat((XRs,XPs),dim=1),projection=False)
        T_s2r = self.TravelTimes(torch.cat((XSs,XRs),dim=1)[0,:][None,:],projection=False)

        dP    = torch.autograd.grad(outputs=(T_s2p+T_r2p), inputs=XPs, grad_outputs=torch.ones((T_s2p+T_r2p).size()).to(torch.device(self.Params['Device'])), 
                                      only_inputs=True,create_graph=False,retain_graph=False)[0]
        dP    = torch.sum(abs(dP),dim=1)

        return T_s2r,torch.cat(((T_s2p+T_r2p)[:,None],dP[:,None]),dim=1).detach()



