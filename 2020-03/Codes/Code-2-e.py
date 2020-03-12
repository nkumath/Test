# Code for issue 2-e.

# import packages
import sklearn
import matplotlib.pyplot as plt
import nltk
import numpy as np
import scipy
import pandas as pd
import xlrd
import xlwt
import math
# from textblob import TextBlob
import textblob
import string

# Name and data path.
# name="microwave"
# name="hair_dryer"
name="pacifier"

path="..\\Data\\"+name+".tsv"

# Read the data.
data=pd.read_csv(path, sep='\t', header=0)
npdata=np.array(data)
length=npdata.shape[0]

# The record of Usefulness Rating.
usefulness=np.array(npdata[:,8:10])

useful=np.array(npdata[:,8])
total=np.array(npdata[:,9])
tempUseful=np.array(npdata[:,9])

def sort(array,length):
    for i in range(0,length):
        for j in range(0,length-1):
            if (array[j]>array[j+1]):
                temp=array[j]
                array[j]=array[j+1]
                array[j+1]=temp
    return array


# np.sort(tempUseful,kind="quicksort")
sort(tempUseful,length)

middle=tempUseful[math.floor(length/2)]
totalNum=0.0
mini=int(tempUseful[0])
maxn=int(tempUseful[length-1])

rating=np.empty(length,dtype=float)

for i in range(0,length):
    rating[i]=0.0
    if useful[i]!=np.nan:
        if useful[i]!=0:
            totalNum+=int(useful[i])
            rating[i]=int(useful[i])/int(total[i])
        if useful[i]==0:
            if total[i]!=0:
                rating[i]=0.012345

totalNum=math.floor(float(totalNum)/float(length))

middle=math.floor((middle+totalNum)/2)


# Function alpha.
rateAlpha=np.empty(length,dtype=float)
def alpha(rating,mini,middle,maxn,rateAlpha,length):
    left=math.exp(0.9)
    right=math.exp(1.1)
    center=math.exp(1)
    for i in range(0,length):
        if rating[i]<middle:
            rateAlpha[i]=math.log(left+(center-left)/(middle-mini)*(rating[i]-mini))*rating[i]
        else:
            rateAlpha[i]=math.log(center+(right-center)/(maxn-middle)*(rating[i]-middle))*rating[i]
    return rateAlpha

def markReviews(pos,sub,score,x,y):
    if pos>=0:
        score=pow(x,pos)/(y*(sub+1))
    else:
        score=pos*pow(x,pos)/(abs(pos)*y*(sub+1))
    return score

rateAlpha=alpha(rating,mini,middle,maxn,rateAlpha,length)


e2excelOutput=np.empty((length,11))
for i in range(0,length):
    for j in range(0,11):
        e2excelOutput[i,j]=0

for i in range(0,length):
    e2excelOutput[i,0]=int(npdata[i,7])
    e2excelOutput[i,1]=rateAlpha[i]
    titleBlob=textblob.TextBlob(str(npdata[i,12]))
    e2excelOutput[i,2]=titleBlob.sentiment.polarity
    e2excelOutput[i,3]=titleBlob.sentiment.subjectivity
    contentBlob=textblob.TextBlob(str(npdata[i,13]))
    e2excelOutput[i,4]=contentBlob.sentiment.polarity
    e2excelOutput[i,5]=contentBlob.sentiment.subjectivity
    e2excelOutput[i,6]=e2excelOutput[i,2]*0.2+e2excelOutput[i,4]*0.8
    e2excelOutput[i,7]=e2excelOutput[i,3]*0.2+e2excelOutput[i,5]*0.8
    temp1=0
    temp2=0
    temp3=0
    e2excelOutput[i,10]=(markReviews(e2excelOutput[i,2],e2excelOutput[i,3],temp1,3,0.5)*0.2+markReviews(e2excelOutput[i,4],e2excelOutput[i,7],temp2,3,0.5)*0.8)
    temp3=temp1*0.2+temp2*0.8
    if e2excelOutput[i,10]>1:
        e2excelOutput[i,8]=e2excelOutput[i,1]
    else:
        e2excelOutput[i,9]=e2excelOutput[i,1]

outputExcelDataFrame=pd.DataFrame(e2excelOutput)
outputExcelDataFrame.columns = ["Star","Usefulness","Title Polarity","Title Subjectivity","Content Polarity","Content Subjectivity","Review Polarity","Review Subjectivity","Positive Usefulness","Negative Usefulness","Review Score"]
writer=pd.ExcelWriter("../"+name+"/2e/For-2e-"+name+".xlsx")
outputExcelDataFrame.to_excel(writer, float_format='%.5f')
writer.save()

print(0)