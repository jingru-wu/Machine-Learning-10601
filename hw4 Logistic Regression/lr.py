from math import *
import numpy as np
import sys
import time

def spatialdot(X,Y):
    s=0
    index=(Y!=0)&(X!=0)
    x=X[index]
    y=Y[index]
    for i in range(len(y)):
        s=s+x[i]*y[i]
    return s

def predication(dataset,theta,labels_out):
    M=len(theta)-1
    X,Y_valid=bulidX_Y(dataset,M)
    thetaT_X=np.dot(X,np.array([theta]).T)
    error=0
    for i in range(len(dataset)):
        exp_thetaT_x=exp(thetaT_X[i])
        pro=exp_thetaT_x/(1+exp_thetaT_x)
        if pro>0.5: label=1
        elif pro<=0.5: label=0
        if Y_valid[i]!=label:
            error=error+1
        str_label=str(label)+'\n'
        labels_out.writelines(str_label)
    return error/len(dataset)




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
    formatted_train= open('E:/ML/HW/HW4/release/handout/largeoutput/model1_formatted_train.tsv','r')
    formatted_valid = open('E:/ML/HW/HW4/release/handout/largeoutput/model1_formatted_valid.tsv','r')
    formatted_test = open('E:/ML/HW/HW4/release/handout/largeoutput/model1_formatted_test.tsv','r')
    dict_file = open('release/handout/dict.txt','r')
    train_out_labels=open('train_out.labels.txt','w')
    test_out_labels=open('test_out.labels.txt','w')
    matrics_out=open('matrices_out.txt','w')

    # interation=60



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
    for k in range(interation):
        print('ineration:',k)
        for i in range(N):
            XI=X[i,:]
            exp_the_Xi=exp(spatialdot(theta,XI))
            S=Y[i]-(exp_the_Xi/(1+exp_the_Xi))
            J_index=XI!=0
            theta[J_index]=theta[J_index]+lamb*S*XI[J_index]
    end1 = time.time()
    print('train time:',end1-start)

    train_error=predication(train_data,theta,train_out_labels)
    # predication(valid_data,theta)
    test_error=predication(test_data,theta,test_out_labels)

    str1='error(train):'+str(train_error)+'\n'
    str2='error(test):'+str(test_error)
    matrics_out.writelines(list(str1))
    matrics_out.writelines(list(str2))

    end2=time.time()
    print('finish time:',end2-start)
