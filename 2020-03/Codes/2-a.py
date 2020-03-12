# Code for issue 2-a.

# improt packages.
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

# Functions for calcuate.

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

def markReviews(pos,sub,score,length,x,y):
    for i in range(0,length):
        if pos[i]>=0:
            score[i]=pow(x,pos[i])/(y*(sub[i]+1))
        else:
            score[i]=pos[i]*pow(x,pos[i])/(abs(pos[i])*y*(sub[i]+1))

def calcuateReviewMarks(data,dataLength,titlePos,contentPos,size,titleMarks,contentMarks):
    titleMarks=np.empty(size,dtype=float)
    contentMarks=np.empty(size,dtype=float)
    temp=np.empty(size,dtype=float)
    titlePositive=np.empty(dataLength,dtype=float)
    titleSub=np.empty(dataLength,dtype=float)
    contentPositive=np.empty(dataLength,dtype=float)
    contentSub=np.empty(dataLength,dtype=float)
    for i in range(0,length):
        temp=0
        titleTemp=textblob.TextBlob(str[data[i,titlePos]])
        titlePositive[i]=temp.sentiment.polarity
        titleSub[i]=temp.sentiment.subjectivity
        contentTemp=textblob.TextBlob(str(data[i,contentPos]))
        contentPositive[i]=temp.sentiment.polarity
        contentSub[i]=temp.sentiment.subjectivity
    markReviews(titlePositive,titleSub,titleMarks,size,3,0.5)
    markReviews(contentPositive,contentSub,contentMarks,size,3,0.5)
    for i in range(0,size):
        temp[i]=titleMarks[i]*0.2+contentMarks[i]*0.8
    return temp

def calcuateHelpfulness(data,dataLength,helpVotesPos,totalVotesPos,size,helpVotes,totalVotes):
    temp=np.empty(dataLength,dtype=float)
    if helpVotes==0 or totalVotes==0:
        for i in range(0,dataLength):
            temp[i]=float(data[i,helpVotesPos])/float(data[i,totalVotesPos])
    else:
        for i in range(0,dataLength):
            temp[i]=float(helpVotes[i])/float(totalVotes[i])
    cache=np.copy(temp)
    sort(temp,dataLength)
    mini=cache[0]
    maxn=cache[dataLength-1]
    middle=cache[dataLength/2]
    alpha(temp,mini,middle,maxn,cache,dataLength)
    return cache

def calcuate(data,dataLength,position,size):
    temp=np.empty(size,dtype=float)
    for i in range(0,length):
        temp[i]=0
        if position==-1:
            temp[int(round(data[i]))]+=1
        else:
            temp[int(round(data[i,position]))]+=1
    return temp

def sort(array,length):
    for i in range(0,length):
        for j in range(0,length-1):
            if (array[j]>array[j+1]):
                temp=array[j]
                array[j]=array[j+1]
                array[j+1]=temp
    return array