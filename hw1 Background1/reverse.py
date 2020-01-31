from __future__ import print_function
import sys

if __name__ == '__main__':
    infile=open(sys.argv[1],'r')
    outfile=open(sys.argv[2],'w')
    # infile=open('example.txt','r')
    # outfile=open('output.txt','w')
    # data=infile.read()
    # infile.close()
    #
    # data_split=data.split('\n')
    # data_reverse=data_split[::-1]
    # data_reverse='\n'.join(data_reverse)
    # #
    # outfile.write(data_reverse)
    # outfile.close()


    for line in reversed(list(infile)):
        print(line)
        outfile.writelines(line)
    outfile.close()

