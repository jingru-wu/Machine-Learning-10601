import math
import numpy as np
import csv
import sys

class Node:
  def __init__(self,l_label='',r_label='',value=None,attr_name=None,l_dataset='',r_dataset='',left=None,right=None,depth=-1,leaf=-1):
    self.attr_name =attr_name
    self.r_label=r_label
    self.l_label=l_label
    self.l_dataset=l_dataset
    self.r_dataset=r_dataset
    self.left=left
    self.right=right
    self.depth=depth
    self.value=value
    self.leaf=leaf

def bulid_tree(node,maxdepth,depth=-1):
    if node.depth>=maxdepth:
        node.leaf=1
        return
    else:
        if node.left is None:
            labels=node.l_dataset[-1]
            u_label,_ = np.unique(labels[1:], return_inverse=True)
            if len(node.l_dataset)==1 or len(u_label)!=2:
                node_leaf=Node(attr_name=node.attr_name,leaf=1,value=node.l_dataset,depth=depth,l_label=node.l_label)
                node.left=node_leaf
            else:
                node.left=stump(node.l_dataset,depth+1)
                bulid_tree(node.left,maxdepth,node.left.depth)
        if node.right is None:
            labels=node.r_dataset[-1]
            u_label,_ = np.unique(labels[1:], return_inverse=True)
            if len(node.r_dataset)==1 or len(u_label)!=2:
                node_leaf=Node(attr_name=node.attr_name,leaf=1,value=node.r_dataset,depth=depth,r_label=node.r_label)
                node.right=node_leaf
            else:
                node.right=stump(node.r_dataset,depth+1)
                bulid_tree(node.right,maxdepth,node.right.depth)

def print_tree(node,label1,label2):
    if node.left!=None:
        cnt1,cnt2=majority_vote(node.l_dataset,label1,label2)
        print(str('|'*(node.depth+1)),node.attr_name,'=:',node.l_label,'[',cnt1,label1,'/',cnt2,label2,']')
        if node.left.leaf!=1:
            print_tree(node.left,label1,label2)
    if node.right!=None:
        cnt1,cnt2=majority_vote(node.r_dataset,label1,label2)
        print(str('|'*(node.depth+1)),node.attr_name,'=:',node.r_label,'[',cnt1,label1,'/',cnt2,label2,']')
        if node.right.leaf!=1:
            print_tree(node.right,label1,label2)
    else:
        return

def search(node,attrs_values,attr_names,label1,label2):
    if node.leaf==1:
        cnt1,cnt2=majority_vote(node.value,label1,label2)
        if cnt1>=cnt2:
            return label1
        else:
            return label2
    else :
        attr_value=attrs_values[attr_names==node.attr_name]
        if node.l_label==attr_value[0]:
            label=search(node.left,attrs_values,attr_names,label1,label2)
            return label
        elif node.r_label==attr_value[0]:
            label=search(node.right,attrs_values,attr_names,label1,label2)
            return label

def majority_vote(dataset,label1,label2):
    labels=dataset[-1]
    labels=np.asarray(labels[1:])
    label=''
    cnt1=sum(labels==label1)
    cnt2=sum(labels==label2)
    # if cnt1==max(cnt1,cnt2):
    #     label=label1
    # elif cnt2==max(cnt1,cnt2):
    #     label=label2
    return cnt1,cnt2

def stump(dataset,depth):
# input: dataset with titile and label
# get attrs, labels, attrs name

    ori_data=dataset.copy()
    labels=dataset[-1]
    attrs=dataset[:-1]
#  select which attr to split
    MI_list=[]
    # get Mutual information of every
    for i in range(len(attrs)):
        a=attrs[i]
        MI=mutual_information(labels[1:],a[1:])
        MI_list.append(MI)
    max_MI=max(MI_list)
    Attr_index=MI_list.index(max_MI)
# get index of this attribute positive and negative
    attr_value=attrs[Attr_index]
    attr_value=np.asarray(attr_value)
    u_attr,indices = np.unique(attr_value[1:], return_inverse=True)
# delete this atrr data
# split rest dataset into positive group and negative group datasets
    l_label=u_attr[1]
    r_label=u_attr[0]
    l_index=np.insert(indices,0,1).astype(bool)
    r_index=np.insert(~indices+2,0,1).astype(bool)
    l_dataset=[]
    r_dataset=[]
    for j in range(len(dataset)):
        col=dataset[j]
        col=np.asarray(col)
        l_col=col[l_index]
        r_col=col[r_index]
        l_dataset.append(list(l_col))
        r_dataset.append(list(r_col))
    del(l_dataset[Attr_index])
    del(r_dataset[Attr_index])
    node=Node(l_label=l_label,r_label=r_label,attr_name=attr_value[0],l_dataset=l_dataset,r_dataset=r_dataset,value=ori_data,depth=depth)
    return node

def get_dataset(argv):
    csvfile=open(argv,newline='')
    reader = csv.reader(csvfile)
    data=[]

    for row in reader:
        data.append(row)
    col_num=len(data[0])
    row_num=len(data)
    column=['1']*row_num
    column_list=[column]*col_num
    for i in range(0,col_num):
        column=['1']*row_num
        for j in range(0,row_num):
            row=data[j]
            column[j] = row[i]
        column_list[i]=column
    label=column_list[-1]
    attr=column_list[:-1]
    return column_list

def get_dataset_rows(argv):
    csvfile=open(argv,newline='')
    reader = csv.reader(csvfile)
    data=[]
    for row in reader:
        data.append(row)
    return data

def mutual_information(Y,X):
    num=len(Y)
    Y=np.asarray(Y)
    uy,_ = np.unique(Y, return_inverse=True)
    p1=sum(Y==uy[0])/num
    p2=sum(Y==uy[1])/num
    Hy=-(p1*math.log2(p1)+p2*math.log2(p2))
    X=np.asarray(X)
    ux, indices = np.unique(X, return_inverse=True)
    H_yx=0
    for i in range(0,len(ux)):
        px=sum(X==ux[i])/num
        Yxv=Y[indices==i]
        P_y1xv=sum(Yxv==uy[0])/sum(X==ux[i])
        P_y2xv=sum(Yxv==uy[1])/sum(X==ux[i])
        if P_y1xv==0: H1=0
        else:H1=-P_y1xv*math.log2(P_y1xv)
        if P_y2xv==0: H2=0
        else:H2=-P_y2xv*math.log2(P_y2xv)
        H_yxv=px*(H1+H2)
        H_yx=H_yx+H_yxv
    I=Hy-H_yx
    return I

def predict(file,label1,label2):
    data_test=get_dataset_rows(file)
    data_num=len(data_test)
    # print(node_root.attr_name,node_root.l_label)
    attr_names=data_test[0]
    attr_names=attr_names[:-1]
    predict_list=[]
    for i in range(1,data_num):
        attrs_values=data_test[i]
        attrs_values=attrs_values[:-1]
        attrs_values=np.asarray(attrs_values)
        attr_names=np.asarray(attr_names)
        predict_label=search(node=node_root,attrs_values=attrs_values,attr_names=attr_names,label1=label1,label2=label2)
        predict_list.append(predict_label)

    return predict_list

## command agrv: max-depth
#                file
if __name__ == '__main__':
    A=['1','1','1','1','0','0','0']
    Y=['0','1','1','1','0','0','0']
    B=['1','1','0','1','0','1','0']
    C=['0','2','0','2','2','1','0']
    # print(mutual_information(Y,A))
    # print(mutual_information(Y,B))
    # print(mutual_information(Y,C))
    # arv1='small_train.csv'
    # arv2='small_test.csv'
    # arv3='2'
    # arv4='small_train_labels.txt'
    # arv5='small_test_labels.txt'
    # arv6='small_matrices.txt'
    # train_file=arv1
    # test_file=arv2
    # d=arv3
    # train_out=open(arv4,'w')
    # test_out=open(arv5,'w')
    # matrics_out=open(arv6,'w')


    train_file=sys.argv[1]
    test_file=sys.argv[2]
    d=sys.argv[3]
    train_out=open(sys.argv[4],'w')
    test_out=open(sys.argv[5],'w')
    matrics_out=open(sys.argv[6],'w')

    max_depth=int(d)
    dataset=get_dataset(train_file)

    labels=dataset[-1]
    u_attr,_ = np.unique(labels[1:], return_inverse=True)
    label1=u_attr[0]
    label2=u_attr[1]

    if max_depth==0:
        cnt1,cnt2=majority_vote(dataset,label1,label2)
        print(cnt1,label1,'/',cnt2,label2)
    else:
        node_root=stump(dataset,depth=0)
        cnt1,cnt2=majority_vote(node_root.value,label1,label2)
        print('[',cnt1,label1,'/',cnt2,label2,']')
        bulid_tree(node_root,maxdepth=max_depth,depth=0)
        print_tree(node_root,label1,label2)

#  predict:
#     input:dataset

    test_predict=np.asarray(predict(test_file,label1,label2))
    train_predict=np.asarray(predict(train_file,label1,label2))
    for i in range(len(test_predict)):
        str1=str(test_predict[i])+'\n'
        test_out.writelines(list(str1))
    for i in range(len(train_predict)):
        str1=str(train_predict[i])+'\n'
        train_out.writelines(list(str1))


    test_label=get_dataset(test_file)
    test_label=np.asarray(test_label[-1])
    test_error=1-sum(test_label[1:]==test_predict)/len(test_predict)

    train_label=get_dataset(train_file)
    train_label=np.asarray(train_label[-1])
    train_error=1-sum(train_label[1:]==train_predict)/len(train_predict)
    str1='error(train):'+str(train_error)+'\n'
    str2='error(test):'+str(test_error)
    matrics_out.writelines(list(str1))
    matrics_out.writelines(list(str2))
    train_out.close()
    test_out.close()
    matrics_out.close()
