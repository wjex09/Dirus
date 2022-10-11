#!/usr/bin/env python3

# Learning XOR function

import numpy as np 
import matplotlib as plt 

train = np.array([0,0],[0,1],[1,0],[1,1])
test = np.array([0],[1],[1],[0])


# This should not work because of linearity and im so fucking stupid

class Perceptron:  
    def __init__(self,train,test,lr=0.01,nodes=2): 
        self.train = train 
        self.test = test 
        self.lr = lr 
        self.nodes = nodes  

        # weights and bias 

        self.w = np.random.uniform(size = nodes) 
        self.b = -1 


    def forward():     
        return 




