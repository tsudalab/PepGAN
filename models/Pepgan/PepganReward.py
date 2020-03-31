from __future__ import print_function
from subprocess import Popen, PIPE
from math import *
import random,os
import random as pr
from copy import deepcopy
import itertools
import time
import math
import argparse
import subprocess
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from mpi4py import MPI
import sys,os
from multiprocessing import Pool
from random import sample
import numpy as np


def redistribution(idx, total):
    idx = (idx + 0.0) / (total + 0.0) * 16.0
    return (np.exp(idx - 8.0) / (1.0 + np.exp(idx - 8.0)))


def rescale(reward):
    reward = np.array(reward)
    x, y = reward.shape
    ret = np.zeros((x, y))
    for i in range(x):
        l = reward[i]
        rescalar = {}
        for s in l:
            rescalar[s] = s
        idxx = 1
        min_s = 1.0
        max_s = 0.0
        for s in rescalar:
            rescalar[s] = redistribution(idxx, len(l))
            #print(rescalar[s])
            idxx += 1
        for j in range(y):
            ret[i, j] = rescalar[reward[i, j]]
            #print(ret[i, j])
    return ret

# classification subroutine
def critic(criticmod,intseq):
    aalen=56
    aalist=['b','a','r','n','d','c','q','e','g','h','i','l','k','m','f','p','s','t','w','y','v','x',' ']
    cri_seq_out=[]
    for i in range(0,len(intseq)):
        x=np.reshape(intseq[i],(1,len(intseq[i])))
        #print(x)
        x_pad= sequence.pad_sequences(x, maxlen=aalen, dtype='int32',padding='post', truncating='pre', value=22.0)
        predictions=criticmod.predict(x_pad)
        cri_seq_out.append(predictions[0][0][0])

    return cri_seq_out

def loadRNN(path,filename):
    json_file = open(path+"/"+filename+".json","r")
    RNNjson = json_file.read()
    json_file.close()
    loadRNN = model_from_json(RNNjson)
    loadRNN.load_weights(path+"/"+filename+".h5")
    return loadRNN

class Reward(object):
    def __init__(self, model, dis, sess, rollout_num):
        self.model = model
        self.dis = dis
        self.sess = sess
        self.rollout_num = rollout_num
    

    def get_reward(self, input_x):
        rewards = []
        lambda_p=0.5
        
        criticmod=loadRNN("/Users/andrejstucs/Documents/Results/16/Descriminator_hawk/GRURNN_AMP","AMPcls-GRU256-64")
        
        for i in range(self.rollout_num):
            for given_num in range(1, self.model.sequence_length // self.model.step_size):
                real_given_num = given_num * self.model.step_size
                feed = {self.model.x: input_x, self.model.given_num: real_given_num, self.model.drop_out: 1.0}
                samples = self.sess.run(self.model.gen_for_reward, feed)
                cri_seq=critic(criticmod,samples)
                feed = {self.dis.D_input_x: samples}
                ypred_for_auc = self.sess.run(self.dis.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])
                ypred = lambda_p*np.array(ypred)+(1.0-lambda_p)*np.array(cri_seq)
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred
            cri_seq=critic(criticmod,input_x)
            feed = {self.dis.D_input_x: input_x}
            ypred_for_auc = self.sess.run(self.dis.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            ypred = lambda_p*np.array(ypred)+(1.0-lambda_p)*np.array(cri_seq)
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[self.model.sequence_length // self.model.step_size - 1] += ypred

        rewards = np.transpose(np.array(rewards)) / (1.0 * self.rollout_num)
        
        return rewards






