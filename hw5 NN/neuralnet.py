import numpy as np
import sys
import csv

def get_dataset(csvfile):
    reader = csv.reader(csvfile)
    data=[]
    for row in reader:
        data.append(row)
    data_marix=np.asarray(data)
    data=np.zeros(data_marix.shape)
    data[:,:]=data_marix
    train_label=data[:,0]
    train_data=data[:,1:]
    return train_label,train_data

def initial_random(num_input,num_hidden,num_output):
    alpha = np.random.uniform(-0.1,0.1, size=(num_hidden,num_input))
    alpha[:,0]=0
    beta = np.random.uniform(-0.1,0.1, size=(num_output,num_hidden+1))
    return np.asarray(alpha),np.asarray(beta)

def initial_zores(num_input,num_hidden,num_output):
    alpha=np.zeros((num_hidden,num_input))
    beta=np.zeros((num_output,num_hidden+1))
    return alpha,beta

def sigmoid(x):
    res = 1/(1 + np.exp(x))
    return res
def softmax(x):
    Ex = np.exp(x)
    result = (Ex / sum(Ex))
    return result
def linearForword(a,alpha):
    b=alpha.dot(a)
    return b
def linearBackword(a,alpha,gb):
    g_alpha=np.dot(gb,a.T)
    g_a=np.dot(alpha.T,gb)
    return g_alpha,g_a

def crossEntropyForward(y_true,y_pre):
    loss=-sum(np.dot(y_true.T,np.log(y_pre)))
    return loss

def forward(x,y_true,alpha,beta):

    a=linearForword(x,alpha)

    z=sigmoid(a)
    Z=np.ones((z.shape[0]+1,z.shape[1]))
    Z[1:,:]=z
    b=linearForword(Z,beta)
    y_pre=softmax(b)
    J=crossEntropyForward(y_true,y_pre)
    O=[a,Z,b,y_pre,J] # intermediate quantities
    return O

def backword(x,y_true,O,alpha,beta):
    (a,z,b,y_pre,J)=O
    # g_J=1
    # g_y_pre=crossEntropyBackward(y_true,y_pre,J,g_J)
    # g_b=softmaxBackword(b,y_pre,g_y_pre)
    # alternative way to calculate g_b
    g_b=y_pre-y_true

    g_beta,g_z=linearBackword(z,beta,g_b)
    z=z.flatten()
    g_z=g_z.flatten()
    g_a=g_z*z*(1-z)
    g_a=g_a[1:]
    g_a = g_a.reshape((len(g_a),1))
    g_alpha,g_x=linearBackword(x,alpha,g_a)
    return g_alpha,g_beta

def predict(dataset,y_train,alpha,beta):
    all_cross_entropy=0
    predict_label_list=[]

    N=dataset.shape[0]
    for i in range(N):
        x=np.asarray(dataset[i]).T
        X=np.ones(129)
        X[1:]=x.flatten()
        X=X.reshape(129,1)
        y_true=np.zeros((num_labels,1))
        y_true[np.int(y_train[i]),:]=1
        O=forward(X,y_true, alpha, beta)
        (a,z,b,y_pre,J)=O

        all_cross_entropy+=J[0]
        # print(y_pre)
        predict_label_list.append(np.argmax(y_pre))

    return all_cross_entropy/N,predict_label_list

if __name__ == '__main__':
    ## load command
    #  python neuralnet.py
    # argv1='smalltrain.csv'
    # argv2='smalltest.csv'
    # argv3='train_out.labels'
    # argv4='test_out.labels'
    # argv5='metrics_out.txt'
    # argv6=2   # num of epoch
    # argv7=4   # hidden unites
    # argv8=2   # init_flag  1:random  2: zeros
    # argv9=0.1  # learning rate

    train_file=open(sys.argv[1],'r')
    test_file=open(sys.argv[2],'r')
    train_out=open(sys.argv[3],'w')
    test_out=open(sys.argv[4],'w')
    metrics_out=open(sys.argv[5],'w')
    num_epoch=np.int(sys.argv[6])
    num_hidden=np.int(sys.argv[7])
    initial_flag=np.int(sys.argv[8])
    learing_rate=np.float(sys.argv[9])
## read data
    y_train,train_data=get_dataset(train_file)
    y_test,test_data,=get_dataset(test_file)

##train
    N =train_data.shape[0]# number of datapoints
    M =train_data.shape[1]# number of features: 128
    num_labels=10 # subset of 26 letters
    ## initialize
    if initial_flag==1:
        alpha,beta=initial_random(M+1,num_hidden,num_labels)
    elif initial_flag==2:
        alpha,beta=initial_zores(M+1,num_hidden,num_labels)
    loss=0

    for time in range(num_epoch):
        for i in range(N):
            x=np.asarray(train_data[i])
            X=np.ones(129)
            X[1:]=x.flatten()
            X=X.reshape(len(X),1)
            y_true=np.zeros((num_labels,1))
            y_true[np.int(y_train[i]),:]=1
            y_true=y_true.reshape(10,1)
            O=forward(X,y_true, alpha, beta)
            g_alpha,g_beta=backword(X,y_true,O,alpha,beta)
            ## update parameters:
            alpha+=g_alpha*learing_rate
            beta-=g_beta*learing_rate


        # get mean cross entropy of train and test date
        mean_corrs_entropy_train,pre_train = predict(train_data,y_train,alpha,beta)
        mean_corrs_entropy_test,pre_test = predict(test_data,y_test,alpha,beta)
        string='epoch='+str(time+1)+' crossentropy(train):'+str(mean_corrs_entropy_train)+'\n'
        metrics_out.writelines(string)
        string='epoch='+str(time+1)+' crossentropy(test):'+str(mean_corrs_entropy_test)+'\n'
        metrics_out.writelines(string)

    ## wirte labels:
    for j in range(len(pre_train)):
        train_out.writelines(str(pre_train[j])+'\n')
    for j in range(len(pre_test)):
        test_out.writelines(str(pre_test[j])+'\n')

    error_train=sum(pre_train!=y_train)/len(y_train)
    string='error(train): '+str(error_train)+'\n'
    metrics_out.writelines(string)
    error_test=sum(pre_test!=y_test)/len(y_test)
    string='error(test): '+str(error_test)+'\n'
    metrics_out.writelines(string)
