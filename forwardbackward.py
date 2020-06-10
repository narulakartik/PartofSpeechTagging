# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:28:27 2020

@author: narul
"""

import numpy as np
import sys
test_input=sys.argv[1]
index_to_word=sys.argv[2]
index_to_tag=sys.argv[3]
hmmprior=sys.argv[4]
hmmemit=sys.argv[5]
hmmtrans=sys.argv[6]
predicted_file=sys.argv[7]
metric_file=sys.argv[8]


predictedfile=open(predicted_file, 'w')
metrics=open(metric_file, 'w')
def forward(d, tag_length,transition, emission, prior):
    #initialization
    a=len(d)
    p=d[0][0]
    alpha=np.zeros((a, tag_length))
    for j in range(tag_length):
        
        alpha[0][j]=prior[j]*emission[j][p-1]
     
        
    for t in range(1,a):
        m=d[t][0]
        for j in range(tag_length):
              for k in range(tag_length):
                alpha[t][j]+=alpha[t-1][k]*transition[k][j] 
            
            
            
              
              alpha[t][j]*=emission[j][m-1]
            
    return alpha


def forwardlog(d, tag_length, transition, emission, prior):
    a=len(d)
    alpha=np.zeros((a, tag_length))
    for j in range(tag_length):
        alpha[0][j]=np.log(prior[j]*emission[j][0])
    
    for t in range(1,a):
        for j in range(tag_length):
            f=0
            for k in range(tag_length):
                f+=(np.exp(alpha[t-1][k]+np.log(transition[j][k])))
                
            #m=max(f)    
            alpha[t][j]+=(np.log(emission[j][t])+np.log(f))
    return alpha

def backward(d, tag_length, transition, emission, prior):
    a=len(d)
    
    
    #initialization
    beta=np.zeros((a,tag_length))
    for j in range(tag_length):
        beta[a-1][j]=1
    t=a-2
    while t>=0:
        m=d[t+1][0]
        for j in range(tag_length):
            for k in range(tag_length):
                beta[t][j]+=beta[t+1][k]*transition[j][k]*emission[k][m-1]
        t-=1        
    return beta    


def backwardlog(d, tag_length, transition, emission, prior):
    a=len(d)
    beta=np.zeros((a,tag_length))
    t=a-2
    while t>=0:
        for j in range(tag_length):
            f=0
            for k in range(tag_length):
                f+=(np.exp(beta[t+1][k]+np.log(transition[j][k])+np.log(emission[k][t+1])))
            beta[t][j]=np.log(f)    
                
        t-=1        
    return beta    

    

def predict(d1,fb):
    predicted=[]
    for i in (fb):
        predicted.append(np.where(i==max(i))[0])
    return predicted
        
        


word_index=[]
for line in open(index_to_word,'r'):
    line=line.strip()
    word_index.append(line)
w=len(word_index)

tag_index=[]
for line in open(index_to_tag,'r'):
    line=line.strip()
    tag_index.append(line)
t=len(tag_index)


transition=np.genfromtxt(hmmtrans)
emission=np.genfromtxt(hmmemit)   
prior=np.genfromtxt(hmmprior)


data=[]
for line in open(test_input,'r'):
    
    line2=line.split()
    for l in line2:
        a=l.split('_')
        
    data.append(line2)
        
d1=[]
for i in range(len(data)):
 
 d2=[]
 for line in data[i]:
       data[i]=(line.split("_"))
       d2.append(data[i])
 d1.append(d2)          
        


for i in range(len(d1)):
    for j in range(len(d1[i])):
        
        d1[i][j][0]=word_index.index(d1[i][j][0])+1
        d1[i][j][1]=tag_index.index(d1[i][j][1])+1
values=[]
for i in range(len(d1)):
    ff=forward(d1[i], t, transition, emission, prior )
    bb=backward(d1[i], t, transition, emission, prior )
    n=ff*bb
    a=predict(d1[i],n)
    values.append(a)

#averageloglikelihood
s=0
x=[]
for i in range(len(d1)):
    avg=0
    alpha=forward(d1[i], t, transition, emission, prior)
    a=len(d1[i])
    for j in range(t):
        avg+=(alpha[a-1][j])
    b=np.log(avg)
    s+=b
loglikelihood=s/len(d1)
x=len(d1)
ff=forward(d1[x-1], t, transition, emission, prior )
bb=backward(d1[x-1], t, transition, emission, prior )
n=ff*bb

#print(values[0])

#accuracy
accuracy=0
length=0
for i in range(len(d1)):
    for j in range(len(d1[i])):
        length+=1
        if values[i][j][0]==d1[i][j][1]-1:
            accuracy+=1
accuracy=(accuracy/length)


for i in range(len(d1)):
    for j in range(len(d1[i])-1):
       predictedfile.write(word_index[d1[i][j][0]-1])
       predictedfile.write("_")
       
       predictedfile.write(tag_index[values[i][j][0]])
       predictedfile.write(" ")
    predictedfile.write(word_index[d1[i][len(d1[i])-1][0]-1])
    predictedfile.write("_")
       
    predictedfile.write(tag_index[values[i][len(d1[i])-1][0]])
    
    predictedfile.write("\n")

metrics.write("Average Log-Likelihood: ")
metrics.write(str(loglikelihood))
metrics.write("\n")
metrics.write("Accuracy: ")
metrics.write(str(accuracy))
