import torch 
import torch.nn as nn
from collections import namedtuple
from typing import Optional, NamedTuple, Tuple, Any, Sequence
from numpy.core.numeric import outer
# from quad_hover import QuadHover
import numpy as np
import matplotlib.pyplot as plt



class CMA_ES:
    def __init__(
        self, 
        function, 
        N,
        xmean
        ): 
        
         #give weights
        self.N = N
        # self.xmean = xmean

        # if xmean==0:
        #     self.xmean = np.random.uniform(0.2, 0.7, size=(1, self.N)).reshape(-1,1)
        # else:
        self.xmean = xmean # hier staat gewicht van synapse


        self.stopfitness = -1.6
        self.stopeval = 1e3*self.N**2

        self.lamba = int(4 + np.floor(3*np.log(self.N))) ####AAAAAANGEPAAAASTTT LET OP TODO: check dit even yo
        self.sigma = 0.5
        self.mu = np.floor(self.lamba/2)
        self.weights = np.log(self.mu+1) - np.log(np.arange(1, self.mu + 1)).reshape(-1,1)
        self.weights = self.weights/np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2/np.sum(self.weights**2)

        self.cc = 4/(N + 4)
        self.cs = (self.mueff + 2)/(self.N + self.mueff + 3)
        self.mucov = self.mueff
        self.ccov = (1/self.mucov) * 2/(self.N + 1.4)**2 + (1- 1/self.mucov) * ((2*self.mueff - 1)/((self.N + 2 )**2 + 2 * self.mueff))
        self.damps = 1 + 2 * np.max([0., np.sqrt((self.mueff - 1)/(self.N+1)) - 1 ]) + self.cs
        self.pc = np.zeros((self.N, 1))
        self.ps = np.zeros((self.N, 1))
        self.B = np.identity(self.N) 
        self.D = np.identity(self.N)
        self.C = np.dot(np.dot(self.B,self.D),np.transpose(np.dot(self.B,self.D)))
        self.eigeneval = 0.
        self.chiN = np.sqrt(self.N) * (1 - 1/(4*self.N)+ 1 / (21*self.N**2))

        self.arz = np.zeros((int(self.N), self.lamba))
        self.arx = np.zeros((int(self.N), self.lamba))
        self.arfitness = np.zeros((self.lamba))
        self.arindex = np.zeros(self.lamba)

        self.array_plot = []

        self.counteval = 0.

        self.function = function

    def optimize_run(self, runs, div, wx):
        for i in range(runs):
            print('gogogo', i)
            for k in range(int(self.lamba)):
                self.arz[:,k] = np.random.normal(0., 0.1, size=(1,self.N)) #0.333  #does not necessarily have to fit column
                # self.arz[:,k] = np.zeros(self.lamba) + 0.2
                self.arx[:,k] = self.xmean.squeeze() + (self.sigma * (np.dot(np.dot(self.B,self.D),self.arz[:,k].reshape(-1,1)))).squeeze()

                ########### constraint
                self.arx[self.arx > 1.] = 0.99999
                self.arx[self.arx < 0.] = 0.0001

                self.arfitness[k] = self.function(self.arx[:,k], div[0], wx[0]) + self.function(self.arx[:,k], div[1], wx[1]) + self.function(self.arx[:,k], div[2], wx[2]) + self.function(self.arx[:,k], div[3], wx[3]) + self.function(self.arx[:,k], div[4], wx[4]) 
                # self.function (genome nodig, place weights, objective)
                self.counteval = self.counteval + 1 

            # Expressing MOO objectives reward easy when considering that the 3D rates have to be a certain level--> more stable evolution

            ##### Potential changes for MOO or 1+ lambda CMA ES
            # self.arindex = -np.argsort(-self.arfitness) #high to low
            # self.arfitness = -np.sort(-self.arfitness)

            self.arindex = np.argsort(self.arfitness) #low to high
            self.arfitness = np.sort(self.arfitness)

            # self.arfitness, self.arindex = self.arfitness[::-1], self.arindex[::-1]

            self.xmean = np.dot(self.arx[:,[self.arindex[:len(self.weights)]]],self.weights).reshape(self.N,1)
            self.zmean = np.dot(self.arz[:,[self.arindex[:len(self.weights)]]],self.weights).reshape(self.N,1)

            self.ps = (1-self.cs)*self.ps + (np.sqrt(self.cs*(2-self.cs)*self.mueff))*np.dot(self.B,self.zmean)
            self.hsig = int(np.linalg.norm(self.ps)/np.sqrt(1-(1-self.cs)**(2*self.counteval/self.lamba))/self.chiN < 1.5+1/(self.N-0.5))
            self.pc = (1-self.cc)*self.pc + self.hsig * np.sqrt(self.cc*(2.-self.cc)*self.mueff) * np.dot(np.dot(self.B,self.D), self.zmean)

            self.C = (1-self.ccov) * self.C + self.ccov *(1/self.mucov) * \
                    (np.dot(self.pc, np.transpose(self.pc))+ (1 - self.hsig)*self.cc*(2-self.cc) * self.C) \
                        + self.ccov * (1-(1/self.mucov)) * np.dot(np.dot(np.dot(np.dot(self.B, self.D), self.arz[:,[self.arindex[:int(self.mu)]]].squeeze()), np.diag(self.weights.squeeze())), np.transpose(np.dot(np.dot(self.B, self.D), self.arz[:,[self.arindex[:int(self.mu)]]].squeeze() ) ))


            self.sigma = self.sigma * np.exp((self.cs/self.damps)*(np.linalg.norm(self.ps)/self.chiN - 1))

            if self.counteval - self.eigeneval > (self.lamba/self.ccov/self.N/10.) :
                self.eigeneval = self.counteval
                self.C = np.triu(self.C) + np.transpose(np.triu(self.C, 1))
                self.B, self.D = np.linalg.eig(self.C)[1], np.diag(np.linalg.eig(self.C)[0])  #check later if order of eigenvalues matter??????? I don't think so
                self.D = np.diag(   np.linalg.eig(np.sqrt(np.diag(np.linalg.eig(self.D)[0])))[0])

            # if self.arfitness[0]<=self.stopfitness:
                # break

            if self.arfitness[0] == self.arfitness[int(min(1+ np.floor(self.lamba/2), 2+np.ceil(self.lamba/4.)))]:
                self.sigma = self.sigma * np.exp(0.2+self.cs/self.damps)
                print('gp')
            
            print(self.counteval, self.arfitness[0], 'worst :', self.arfitness[-1])
            self.array_plot.append([self.arx[0,self.arindex[0]], self.arx[1,self.arindex[0]]])

        #write weights to pickle
            weights = self.arx
            best_fitness = self.arfitness[0]
            print(best_fitness)
        return weights[:,0], best_fitness

