import sklearn
import matplotlib.pyplot as plt
import nltk
import wordcloud
import scapy
import numpy as np
import scipy
import pandas as pd
import xlrd
import xlwt
name="Microwave"
path="/home/aistudio/data/microwave.tsv"
io='data/microwave.tsv'
data=pd.read_csv(path, sep='\t', header=0)
# star_rating=pd.read_csv(path, sep='\t', header=0, index_col='star_rating')
npdata=np.array(data)
print(npdata.shape)
print(npdata.shape[0])
print(npdata.shape[1])
star=np.array([0,0,0,0,0])
vine=np.array([0,0,0,0,0])
purchase=np.array([0,0,0,0,0])
for i in range(0,npdata.shape[0]):
    star[int(npdata[i:i+1,7:8]-1)]+=1
    if npdata[i:i+1,10:11]=="Y":
        vine[int(npdata[i:i+1,7:8])-1]+=1
    if npdata[i:i+1,11:12]=="Y":
        purchase[int(npdata[i:i+1,7:8])-1]+=1
plt.bar(range(0,5),star)
plt.bar(range(0,5),purchase)
plt.bar(range(0,5),vine)
plt.savefig(name+"Star_rating.png")
plt.clf()
plt.scatter((npdata[:,7:8]),(npdata[:,8:9]))
plt.scatter((npdata[:,7:8]),(npdata[:,9:10]))
mark=np.empty((npdata.shape[0]*1),dtype=int)
starlist=np.empty((npdata.shape[0]*1),dtype=int)
rate=np.empty((npdata.shape[0]*1),dtype=float)
for i in range(0,npdata.shape[0]):
    starlist[i]=int(npdata[i:i+1,7:8])
    mark[i]=int(npdata[i:i+1,8:9])
    if int(npdata[i:i+1,8:9]) == 0:
        continue
    rate[i]=float(npdata[i:i+1,8:9])/float(npdata[i:i+1,9:10])
plt.savefig(name+"MarksAndStar.png")
plt.clf()
plt.scatter(rate,mark)
plt.savefig(name+"RateOfMark.png")
plt.clf()
plt.scatter(starlist,rate)
plt.savefig(name+"StarAndRate.png")
