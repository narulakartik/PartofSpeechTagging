# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:54:49 2020

@author: narul
"""

import numpy as np
import sys

train_input=sys.argv[1]
index_to_word=sys.argv[2]
index_to_tag=sys.argv[3]
hmmprior=sys.argv[4]
hmmemit=sys.argv[5]
hmmtrans=sys.argv[6]



#for line in open('trainwords.txt','r'):
 #   print(line)
    
    
#word_index=[]
#for line in open('index_to_word.txt', 'r'):
 #   word_index.append(line)
    
    
#word_index=np.loadtxt('index_to_word.txt', dtype=str)
#print(word_index[0])

#tag_index=np.loadtxt('index_to_tag.txt', dtype=str)
#print(tag_index)
#f=open('index_to_word.txt', 'r+')
#word_index = [line for line in f.readlines()]
#f.close()
word_index=[]
for line in open(index_to_word,'r'):
    line=line.strip()
    word_index.append(line)
#print(word_index)


tag_index=[]
for line in open(index_to_tag,'r'):
    line=line.strip()
    tag_index.append(line)


#print(word_index.index('The'))
#print(word_index[3071])
data=[]

for line in open(train_input,'r'):
    
    line2=line.split()
    for l in line2:
        a=l.split('_')
        
        
      #  d2.append(b)
    #line3=line2.split("_")
    data.append(line2)
#print(d1[1])
data2=data
#print(data2[1])
#print(data[1])
d1=[]
for i in range(len(data)):
 #data2[i].clear()
 d2=[]
 for line in data[i]:
       data[i]=(line.split("_"))
       d2.append(data[i])
 d1.append(d2)    

for i in range(len(d1)):
    for j in range(len(d1[i])):
        
        d1[i][j][0]=word_index.index(d1[i][j][0])+1
        d1[i][j][1]=tag_index.index(d1[i][j][1])+1
        
#print(d1[1])

# prior
counts=[1 for i in range(len(tag_index))]
#count=np.ones(len(tag_index))
for i in range(len(d1)):
    
    for j in range(len(tag_index)):
        
        if d1[i][0][1]==j+1:
            counts[j]+=1
counts=[counts[j]/(len(d1)+len(tag_index)) for j in range(len(counts))]
prior=np.asarray(counts)
np.savetxt(hmmprior, prior, delimiter='\n')



# transition
#tc=[0 for i in range(len(tag_index))]
tc=np.zeros((len(tag_index)))
transition=np.ones((len(tag_index), len(tag_index)))
for i in range(len(d1)):                         #sentences
    for j in range(len(d1[i])-1):                       #words
         
         
         a=d1[i][j][1]
         
         tc[a-1]+=1
         
         k=j+1
         
         b=d1[i][k][1]
         transition[a-1][b-1]+=1

tc=np.add(tc,len(tag_index))  
tc=np.transpose(tc)    
tc=np.reciprocal(tc)
#tc=np.transpose(tc)
#transition=transition*np.transpose(tc)
#for line in data:

tp=np.transpose(transition)*tc
transition=np.transpose(tp)


np.savetxt(hmmtrans, transition)


#emission

ec=np.zeros((len(tag_index)))
emission=np.ones((len(tag_index), len(word_index)))
for i in range(len(d1)):
    for j in range(len(d1[i])):
        a=d1[i][j][1]
        b=d1[i][j][0]
        ec[a-1]+=1
        emission[a-1][b-1]+=1
        
ec=np.add(ec,len(word_index))

ec=np.transpose(ec)    
ec=np.reciprocal(ec)
ep=np.transpose(emission)*ec
emission=np.transpose(ep)

#emission=emission/ec
np.savetxt(hmmemit, emission)

 #   for l in line.split("_"):
#print(data)
#for line in data:
 #   dict={}
  #  m=line[0].split("_")[1]
   # c.append(tag_index.index(m))
   # for l in line:
    #    a=l.split("_")
        #dict[a[0]]=a[1]
     #   dict[word_index.index(a[0])+1]=tag_index.index(a[1])+1
    #q.append(dict)
#print(q[3])
#print(data[0][0])  
     
#c1,c2,c3,c4,c5,c6,c7,c8,c9=1,1,1,1,1,1,1,1,1
#counter=[]

#for j in range(len(tag_index)):
 #      counter.append(c.count(j))
  #     counter[j]=(counter[j]+1)/(len(c)+len(tag_index))
#c1/=len(c)

#print(counter)

#print(q)


 #c2/=len(c)
#c3/=len(c)
#c4/=len(c)
#c5/=len(c)
#c6/=len(c)
#c7/=len(c)
#c8/=len(c)        
#c9/=len(c)
#print(c1,c2,c3,c4,c5,c6,c7,c8,c9)
#a=[2,3,4]
#print(a[a.index(2)])
#for l in (data[2]):
 #   data[2][data[2].index(l)]=1
#print(len(data[2]))        
#prior probabilities

#c,d=[],[]
#for i in range(len(data)):
 #   a=data[i][0].split("_")
    #c.append(a[0])
  #  d.append(a[1])
#print(d)




#prior
#c1,c2,c3,c4,c5,c6,c7,c8,c9=0,0,0,0,0,0,0,0,0

#for i in range(len(c)):
 #   if c[i]==
    
