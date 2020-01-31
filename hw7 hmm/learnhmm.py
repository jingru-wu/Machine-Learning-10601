"""
python learnhmm.py trainwords.txt index_to_word.txt index_to_tag.txt hmmprior.txt hmmemit.txt hmmtrans.txt
"""
import numpy as np
import sys
def readdata(data_file):
    data=data_file.read()
    data_word=[]
    data_tag=[]
    lines=data.split('\n')
    for line in lines:
        word_line=[]
        tag_line=[]
        word_tag_s=line.split(' ')
        for word_tag in word_tag_s:
            temp=word_tag.split('_')
            if len(temp)==2:
                word=temp[0]
                tag=temp[1]
                word_line.append(word)
                tag_line.append(tag)
        data_word.append(word_line)
        data_tag.append(tag_line)
    return data_word,data_tag

def readindex(index_file):
    data = index_file.read()
    word2index=dict([])
    index2word=dict([])
    lines=data.strip().split('\n')
    for i in range(len(lines)):
        word=lines[i]
        word2index[word]=i
        index2word[i]=word
    return word2index,index2word

def get_matrix(train_word,train_tag):
    num_word=len(word2index)
    num_tag=len(tag2index)
    count_prior,count_emmision,count_transition=get_count(train_word,train_tag,num_word,num_tag)
    emission_matrix=count2probility(count_emmision)
    transition_matrix=count2probility(count_transition)
    prior_matrix=count2probility(count_prior.T)
    return prior_matrix.T,emission_matrix,transition_matrix

def get_count(train_word,train_tag,num_word,num_tag):
    count_emmision=np.ones((num_tag,num_word))
    count_prior=np.ones((num_tag,1))
    count_transition=np.ones((num_tag,num_tag))
    for i in range(len(train_word)):
    # for i in range(10):
        for j in range(len(train_word[i])):
            word=train_word[i][j]
            tag_now=train_tag[i][j]
            index_word_now=word2index[word]
            index_tag_now=tag2index[tag_now]
            count_emmision[index_tag_now,index_word_now]+=1
            if j==0:
                count_prior[index_tag_now,0]+=1
                tag_pre=tag_now
            else:
                index_tag_pre=tag2index[tag_pre]
                count_transition[index_tag_pre,index_tag_now]+=1
                tag_pre=tag_now
    return  count_prior,count_emmision,count_transition

def count2probility(count):
    probility=count.copy()
    for i in range(len(count)):
        probility[i,:]=count[i,:]/sum(count[i,:])
    return probility

def write_matirx(matrix,write_file):
    num_row=len(matrix)
    num_colum=len(matrix[0])
    for i in range(num_row):
        string=''
        for j in range(num_colum):
            pro=str(matrix[i][j])
            string=string+pro
            if j!=num_colum-1:
                string+=' '
        write_file.write(string+'\n')


if __name__ == '__main__':
    argv=['','trainwords.txt','index_to_word.txt','index_to_tag.txt','hmmprior.txt','hmmemit.txt','hmmtrans.txt']
    # argv=['','toytrain.txt','toy_index_to_word.txt','toy_index_to_tag.txt','hmmprior.txt','hmmemit.txt','hmmtrans.txt']

    # train_input = open(sys.argv[1],'r')
    # index_word = open(sys.argv[2],'r')
    # index_tag = open(sys.argv[3],'r')
    # hmm_prior=open(sys.argv[4],'w')
    # hmm_emission=open(sys.argv[5],'w')
    # hmm_transition=open(sys.argv[6],'w')

    train_input = open(argv[1],'r')
    index_word = open(argv[2],'r')
    index_tag = open(argv[3],'r')
    hmm_prior=open(argv[4],'w')
    hmm_emission=open(argv[5],'w')
    hmm_transition=open(argv[6],'w')

    word2index,index2word=readindex(index_word)
    tag2index,index2tag=readindex(index_tag)
    train_word,train_tag=readdata(train_input)

    prior_matrix,emission_matrix,transition_matrix=get_matrix(train_word,train_tag)
    write_matirx(prior_matrix,hmm_prior)
    write_matirx(emission_matrix,hmm_emission)
    write_matirx(transition_matrix,hmm_transition)
