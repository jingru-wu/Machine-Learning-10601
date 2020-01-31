"""
python forwardbackward.py testwords.txt index_to_word.txt index_to_tag.txt hmmprior.txt hmmemit.txt hmmtrans.txt predict.txt metrics.txt
arguments:
<test input>
<index to word>
<index to tag>
<hmmprior>
<hmmemit>
<hmmtrans>
<predicted file>
<metric file>.
"""

from learnhmm import *

def readmatrix(data_input):
    data = data_input.read()
    lines=data.split('\n')
    num_row=len(lines)-1
    num_colum=len(lines[0].split(' '))
    matrix=np.zeros((num_row,num_colum))
    for i in range(num_row):
        row=lines[i].split(' ')
        matrix[i,:]=np.asarray(row)
    return matrix


def normonize(matrix):
    matrix = [(row/np.sum(row)) for row in matrix]
    return matrix
def wirte_prediction(predictions,test_word,predict_file):

    for i in range(len(test_word)):
        string=''
        for j in range(len(test_word[i])):
            word=test_word[i][j]
            predict_tag=predictions[i][j]
            if j==len(test_word[i])-1:
                string+=word+'_'+predict_tag
            else:
                string+=word+'_'+predict_tag+' '
        predict_file.write(string+'\n')

if __name__ == '__main__':
    argv=['','testwords.txt','index_to_word.txt','index_to_tag.txt','hmmprior.txt','hmmemit.txt','hmmtrans.txt','predict.txt','metrics.txt']
    # argv=['','toytest.txt','toy_index_to_word.txt','toy_index_to_tag.txt','hmmprior.txt','hmmemit.txt','hmmtrans.txt','predict.txt','metrics.txt']
    # test_input = open(sys.argv[1],'r')
    # index_word = open(sys.argv[2],'r')
    # index_tag = open(sys.argv[3],'r')
    # hmm_prior=open(sys.argv[4],'r')
    # hmm_emission=open(sys.argv[5],'r')
    # hmm_transition=open(sys.argv[6],'r')
    # predict_file=open(sys.argv[7],'w')
    # metrics_output=open(sys.argv[8],'w')
    #
    #
    test_input = open(argv[1],'r')
    index_word = open(argv[2],'r')
    index_tag = open(argv[3],'r')
    hmm_prior=open(argv[4],'r')
    hmm_emission=open(argv[5],'r')
    hmm_transition=open(argv[6],'r')
    predict_file=open(argv[7],'w')
    metrics_output=open(argv[8],'w')

    tag2index,index2tag = readindex(index_tag)
    word2index,index2word = readindex(index_word)
    test_word,test_tag=readdata(test_input)
    prior_matrix=readmatrix(hmm_prior)
    emission_matrix=readmatrix(hmm_emission)
    transition_matrix=readmatrix(hmm_transition)
    logP_list=[]
    predictions=test_tag.copy()
    count=np.zeros(2)
    for l in range(len(test_word)):
        # print('sentence:',l)
        sentence=test_word[l]
        loss_list=[]
        alpha = []
        beta = []

        for t in range(len(sentence)):
            word = sentence[t]
            word_index=word2index[word]
            bxt = emission_matrix[:,word_index] # K*M
            if t == 0:
                alphat = prior_matrix.flatten() * bxt
            else:
                alphat_1= alpha[t-1].reshape((len(alpha[t-1]),1))  #K*1
                AjAplhat_1=np.dot(transition_matrix.T,alphat_1)     #K*1
                alphat = bxt.flatten() *AjAplhat_1.flatten()

            if t==len(sentence)-1:
                alpha.append(alphat)
                logP = np.log(np.sum(alpha[-1]))
                logP_list.append(logP)
            else:
                # alpha.append(alphat/np.sum(alphat))
                alpha.append(alphat)
        # print(np.asarray(alpha).T)


        for i in range(len(sentence)):
            t=len(sentence)-i   # t: [T....0]

            if t == len(sentence):
                betat = np.ones(prior_matrix.shape).flatten()
            else:
                word = sentence[t]
                index_word=word2index[word]
                bxt_1 = emission_matrix[:,index_word]
                B_beta=beta[i-1].flatten() * bxt_1.flatten()
                B_beta=B_beta.reshape((len(B_beta),1))
                betat = np.dot(transition_matrix,B_beta)
            beta.append(betat.flatten())
            beta2=beta.copy()
            beta2.reverse()
        beta=np.asarray(beta)
        # print(np.asarray(beta2).T)


        # alpha = [(row/np.sum(row)) for row in alpha]
        # beta = [(row/np.sum(row)) for row in beta]


        for t in range(len(sentence)):
            count[1]+=1
            at = alpha[t]
            bt = beta2[t]
            probs = at.flatten() * bt.flatten()
            max_index=np.argmax(probs)
            predict_tag=index2tag[max_index]
            true_tag=test_tag[l][t]
            predictions[l][t]=predict_tag
            if true_tag==predict_tag:
                count[0]+=1

    wirte_prediction(predictions,test_word,predict_file)
    average_logP=np.mean(logP_list)
    string1='Average Log-Likelihood: '+str(average_logP)+'\n'
    string2='Accuracy: '+str(count[0]/count[1])
    metrics_output.write(string1+string2)
    print(string1+string2)
