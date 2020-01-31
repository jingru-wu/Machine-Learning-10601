import math
import numpy as np
import sys

def model2(dataset,dict_list,formatted_file_out):
    for i in range(len(dataset)):
        data=dataset[i]
        words=data[2:]
        words=words.split(' ')
        label=data[0]
        threshold=4
        count=np.zeros(len(dict_list))
        for word in words:
            if word in dict_list:
                index=dict_list[word]
                count[index]=count[index]+1
        l1=0<count
        l2=count<threshold
        count=l1&l2
        print_str=label+'\t'
        for j in range(len(count)):
            if count[j]==True:
                print_str=print_str+str(j)+':1 '
        print_str=list(print_str)
        print_str[-1]='\n'
        print_str=''.join(print_str)
        # print(print_str)
        formatted_file_out.writelines(print_str)

def model1(dataset,dict_list,formatted_file_out):
    for i in range(len(dataset)):
        data=dataset[i]
        words=data[2:]
        words=words.split(' ')
        label=data[0]
        count=np.zeros(len(dict_list))
        for word in words:
            if word in dict_list.keys():
                index=dict_list[word]
                count[index]=count[index]+1
        print_str=label+'\t'
        for j in range(len(count)):
            if count[j]>0:
                print_str=print_str+str(j)+':1\t'
        print_str=list(print_str)
        print_str[-1]='\n'
        print_str=''.join(print_str)
        # print(print_str)
        formatted_file_out.writelines(print_str)



if __name__ == '__main__':
#     train_file = open('train_data.tsv','r')
#     valid_file = open('valid_data.tsv','r')
#     test_file = open('test_data.tsv','r')
#     dict_file = open('dict.txt','r')
# ## output files
#     formatted_train_out= open('formatted_train.tsv','w')
#     formatted_valid_out = open('formatted_valid.tsv','w')
#     formatted_test_out = open('formatted_test.tsv','w')
#     feature_flag=1
    train_file = open(sys.argv[1],'r')
    valid_file = open(sys.argv[2],'r')
    test_file = open(sys.argv[3],'r')
    dict_file = open(sys.argv[4],'r')
    formatted_train_out= open(sys.argv[5],'w')
    formatted_valid_out = open(sys.argv[6],'w')
    formatted_test_out = open(sys.argv[7],'w')
    feature_flag=np.int(sys.argv[8])

    train_data = train_file.readlines()
    valid_data = valid_file.readlines()
    test_data = test_file.readlines()
    dict_data = dict_file.readlines()

    dict_list =dict([])
    for i in range(len(dict_data)):
        data=dict_data[i]
        data=data.split(' ')
        num=np.int(data[1])
        name=data[0]
        dict_list[name]=num

    if feature_flag==1:
        model1(train_data,dict_list,formatted_train_out)
        model1(valid_data,dict_list,formatted_valid_out)
        model1(test_data,dict_list,formatted_test_out)
    if feature_flag==2:
        model2(train_data,dict_list,formatted_train_out)
        model2(valid_data,dict_list,formatted_valid_out)
        model2(test_data,dict_list,formatted_test_out)

    print('finished test')
    formatted_train_out.close()
    formatted_valid_out.close()
    formatted_test_out.close()
