from math import *
import numpy as np
import time
import matplotlib.pyplot as plt
def spatialdot(X,Y):
    s=0
    index=(Y!=0)&(X!=0)
    x=X[index]
    y=Y[index]
    for i in range(len(y)):
        s=s+x[i]*y[i]
    return s

def logloss(true_label, predicted, eps=1e-15):
  p = np.clip(predicted, eps, 1 - eps)
  if true_label == 1:
    return -log(p)
  else:
    return -log(1 - p)

def predication(dataset,theta):
    M=len(theta)-1
    NNL=0
    labels=np.zeros(len(dataset))
    X,Y_valid=bulidX_Y(dataset,M)
    thetaT_X=np.dot(X,np.array([theta]).T)
    error=0
    for i in range(len(dataset)):
        exp_thetaT_x=exp(thetaT_X[i])
        pro=exp_thetaT_x/(1+exp_thetaT_x)
        if pro>0.5: labels[i]=1
        elif pro<=0.5: labels[i]=0
        if Y_valid[i]!=labels[i]:
            error=error+1
        loss=-Y_valid[i]*thetaT_X[i]+log(1+exp_thetaT_x)
        NNL=NNL+loss
    return error/len(dataset),NNL/len(dataset)

def bulidX_Y(dataset,M):
    N=len(dataset)
    Y=np.zeros(N)
    X=np.zeros((N,M+1))
    X[:,0]=1
    for i in range(N):
        data=dataset[i]
        data=data.split('\t')
        Y[i]=np.int(data[0])
        del data[0]
        for j in range(len(data)):
            index=data[j].split(':')
            index=np.int(index[0])
            X[i,index+1]=1


    return X,Y



if __name__ == '__main__':
    start = time.time()
    ## load dataset
    formatted_train= open('release/handout/largeoutput/model1_formatted_train.tsv','r')
    formatted_valid = open('release/handout/largeoutput/model1_formatted_valid.tsv','r')
    formatted_test = open('release/handout/largeoutput/model1_formatted_test.tsv','r')
    dict_file = open('release/handout/dict.txt','r')

    ineration=50
    train_log_likelihood=np.zeros(ineration)
    valid_log_likelihood=np.zeros(ineration)


    # formatted_train= open(sys.argv[1],'r')
    # formatted_valid = open(sys.argv[2],'r')
    # formatted_test = open(sys.argv[3],'r')
    # dict_file = open(sys.argv[4],'r')
    # train_out_labels=open(sys.argv[5],'w')
    # test_out_labels=open(sys.argv[6],'w')
    # matrics_out=open(sys.argv[7],'w')
    # interation=np.int(sys.argv[8])

    train_data=formatted_train.read()
    valid_data=formatted_valid.read()
    dict_data = dict_file.readlines()
    test_data=formatted_test.read()

    train_data=train_data.split('\n')
    test_data=test_data.split('\n')
    valid_data=valid_data.split('\n')
    del train_data[-1]
    del valid_data[-1]
    del test_data[-1]

    N=len(train_data)
    M=len(dict_data)

    lamb= 10e-2

    # build X matrix
    X,Y=bulidX_Y(train_data,M)

    ## train
    theta=np.zeros(M+1)
    NLL_train=np.zeros(ineration)
    NLL_valid=np.zeros(ineration)
    for k in range(ineration):
        print('ineration:',k)
        for i in range(N):
            XI=X[i,:]
            exp_the_Xi=exp(spatialdot(theta,XI))
            S=Y[i]-(exp_the_Xi/(1+exp_the_Xi))
            J_index=XI!=0
            theta[J_index]=theta[J_index]+lamb*S*XI[J_index]

        train_error,NLL_train[k]=predication(train_data,theta)
    # predication(valid_data,theta)
        test_error,NLL_valid[k]=predication(test_data,theta)
    print(train_error)
    print(test_error)
    #     print('train:',NLL_train[k])
    #     print('valid:',NLL_valid[k])
    # train_nll, = plt.plot(NLL_train)
    # valid_nll, = plt.plot(NLL_valid)
    # plt.legend([train_nll, valid_nll], ['Train NLL', 'Valid NLL'])
    # plt.ylabel('Negative Log-Likelihood')
    # plt.xlabel('Epochs')
    # plt.show()
