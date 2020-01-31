import csv
import sys
import math
import numpy as np

if __name__ == '__main__':
    csvfile=open(sys.argv[1],'r')
    outfile=open(sys.argv[2],'w')
# read data in columns
    reader = csv.reader(csvfile)
    data=[]
    data_num=[]
    column_list=[]
    for row in reader:
        data.append(row)
    col_num=len(data[0])
    row_num=len(data)
    column=['1']*row_num

    for i in range(0,col_num):
        for j in range(0,row_num):
            row=data[j]
            column[j] = row[i]
        # print(column)
        column_list.append(column)
label=column_list[-1]
label=label[1:]
# calculate entropy and error
Y=np.asarray(label)
uy,_ = np.unique(Y, return_inverse=True)
cnt1=sum(Y==uy[0])
cnt2=sum(Y==uy[1])
p1=cnt1/len(label)
p2=cnt2/len(label)
error=min(p1,p2)
entropy=-(p1*math.log2(p1)+p2*math.log2(p2))

str1='entropy: '+str(entropy)+'\n'
str2='error: '+str(error)+'\n'
outfile.writelines(str1)
outfile.writelines(str2)

csvfile.close()
outfile.close()
