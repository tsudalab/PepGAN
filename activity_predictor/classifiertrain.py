import numpy as np
import os
import sys
import multiprocessing as mp
from keras.models import Sequential
from keras.layers import Dense, Activation,TimeDistributed,MaxPooling1D
from keras.layers import LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.layers import Dropout
import numpy as np
import random
import sys
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import model_from_json
from random import sample
from keras.callbacks import CSVLogger

#Defining the vocabulary (should be constant with the vocabulary used for the generative model)
aalist=["B","A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V","X"," "]

def seqfrmat(seqinp,maxlnpep):
    tmp=seqinp.strip()+"X"
    while len(tmp)<=maxlnpep:
        tmp=tmp+" "
    coding=[]
    seqid=[]
    for x in range(0,maxlnpep+1):
        tmpctgr=to_categorical(aalist.index(tmp[x]), num_classes=len(aalist))
        coding.append(tmpctgr) 
        seqid.append(aalist.index(tmp[x]))
    return seqid,coding

def loaddata(csvpath,csvpathneg,maxlnpep):
    f=open(csvpath,'r')
    ln=f.readlines()[1:]
    lenln=len(ln)
    clnpep=[]
    clncoding=[]
    f.close()  
    fn=open(csvpathneg,'r')
    lnn=fn.readlines()[1:]
    lenlnn=len(lnn)
    fn.close()
    datacutoff=0
    f=open("RNN-dropoutdata-GRU256-64.csv","w")
    seqlist=sample(range(0,lenln),lenln)  
    seqlistneg=sample(range(0,lenlnn),lenlnn)
    for i in range(0,lenln):
        if (len(ln[i])<=maxlnpep)&(i in seqlist):
            frmseq,frmcod=seqfrmat(ln[i],maxlnpep)
            frmcod=[[1]]
            clnpep.append(frmseq)
            clncoding.append(frmcod)
        else:
            f.write(ln[i].strip()+"X"+"\n")
    for i in range(0,lenlnn):
        if (len(lnn[i])<=maxlnpep)&(i in seqlistneg):
            frmseq,frmcod=seqfrmat(lnn[i],maxlnpep)
            frmcod=[[0]]
            clnpep.append(frmseq)
            clncoding.append(frmcod)
        else:
            f.write(lnn[i].strip()+"X"+"\n")
    f.close()
    return clnpep,clncoding

def save_model(model):
    model_json = model.to_json()
    with open("Model-GRU256-64.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("Model-GRU256-64.h5")
    print("Saved model to disk")

if __name__ == "__main__":
    maxlnpep=55
    nproc=4
    #Set the link to the positive and negative data
    Positive_set,Negative_set=loaddata("/Users/andrejstucs/Documents/Results/16/PepGAN/data/amp_all.csv","/Users/andrejstucs/Documents/Results/16/PepGAN/data/nonamp.csv",maxlnpep)

    X=np.array((Positive_set))
    Y=np.array((Negative_set))
    model = Sequential()
    aalstln=len(aalist)
    dataln=X.shape[1]
    #Model set-up
    model.add(Embedding(input_dim=aalstln, output_dim=len(aalist), input_length=dataln,mask_zero=False))
    model.add(GRU(output_dim=256, activation='tanh',return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid')) 
    model.add(MaxPooling1D(pool_size=52))
    optimizer=Adam(lr=0.00001)
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history_callback = model.fit(X,Y,epochs=1000, batch_size=512,validation_split=0.1)
    loss_history = history_callback.history["loss"]
    acc_history = history_callback.history["acc"]
    val_loss_history = history_callback.history["val_loss"]
    val_acc_history = history_callback.history["val_acc"]
    numpy_loss_history = np.array(loss_history)
    numpy_acc_history = np.array(acc_history)
    numpy_val_loss_history = np.array(val_loss_history)
    numpy_val_acc_history = np.array(val_acc_history)
    np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")
    np.savetxt("acc_history.txt", numpy_acc_history, delimiter=",")
    np.savetxt("val_loss_history.txt", numpy_val_loss_history, delimiter=",")
    np.savetxt("val_acc_history.txt", numpy_val_acc_history, delimiter=",")
    save_model(model) 



