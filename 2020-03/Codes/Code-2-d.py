# Code for issue 2-d.

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

# Name and data path.
name="microwave"
# name="hair_dryer"
# name="pacifier"

path="..\\Data\\"+name+"2a.xlsx"


# Functions for calculate.

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

def beta(mini,middle,maxn,length,rating):
    left = math.exp(0.9)
    right = math.exp(1.1)
    center = math.exp(1)
    for i in range(0, length):
        if rating[i] < middle:
            rateAlpha[i] = math.log(left + (center - left) / (middle - mini) * (rating[i] - mini))
        else:
            rateAlpha[i] = math.log(center + (right - center) / (maxn - middle) * (rating[i] - middle))
    return rateAlpha

def markReviews(pos,sub,score,length,x,y):
    for i in range(0,length):
        if pos[i]>=0:
            score[i]=pow(x,pos[i])/(y*(sub[i]+1))
        else:
            score[i]=pos[i]*pow(x,pos[i])/(abs(pos[i])*y*(sub[i]+1))
    return score

def calcuateReviewMarks(data,dataLength,titlePos,contentPos,size,titleMarks,contentMarks):
    titleMarks=np.empty(size,dtype=float)
    contentMarks=np.empty(size,dtype=float)
    temp=np.empty(size,dtype=float)
    titlePositive=np.empty(dataLength,dtype=float)
    titleSub=np.empty(dataLength,dtype=float)
    contentPositive=np.empty(dataLength,dtype=float)
    contentSub=np.empty(dataLength,dtype=float)
    for i in range(0,length):
        temp[i]=0
        titleTemp=textblob.TextBlob(str(data[i,titlePos]))
        titlePositive[i]=titleTemp.sentiment.polarity
        titleSub[i]=titleTemp.sentiment.subjectivity
        contentTemp=textblob.TextBlob(str(data[i,contentPos]))
        contentPositive[i]=contentTemp.sentiment.polarity
        contentSub[i]=contentTemp.sentiment.subjectivity
    titleMarks=markReviews(titlePositive,titleSub,titleMarks,size,3,0.5)
    contentMarks=markReviews(contentPositive,contentSub,contentMarks,size,3,0.5)
    for i in range(0,size):
        temp[i]=titleMarks[i]*0.2+contentMarks[i]*0.8
    return temp,titleMarks,contentMarks

def calcuateHelpfulness(data,dataLength,helpVotesPos,totalVotesPos,size,helpVotes,totalVotes):
    temp=np.empty(dataLength,dtype=float)
    for i in range(0,dataLength):
        temp[i]=0
    if helpVotes==0 or totalVotes==0:
        for i in range(0,dataLength):
            if int(data[i,totalVotesPos])!=0:
                if int(data[i,helpVotesPos])==0:
                    temp[i]=0.00001
                else:
                    temp[i]=float(data[i,helpVotesPos])/float(data[i,totalVotesPos])
    else:
        for i in range(0,dataLength):
            if totalVotes[i]!=0:
                if helpVotes[i]==0:
                    temp[i]=0.00001
                else:
                    temp[i]=float(helpVotes[i])/float(totalVotes[i])
    cache=np.copy(temp)
    sort(cache,dataLength)
    mini=cache[0]
    maxn=cache[dataLength-1]
    middle=cache[int(dataLength/2)]
    return alpha(temp,mini,middle,maxn,cache,dataLength)

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

# Read the data.
data=pd.read_excel(path, sep='\t', header=0)
npdata=np.array(data)
length=npdata.shape[0]

reviewTitle=np.empty(length,dtype=float)
reviewContent=np.empty(length,dtype=float)
reviewForExcel=np.empty(length,dtype=float)
helpfulnessForExcel=np.empty(length,dtype=float)
for i in range(0,length):
    reviewTitle[i]=0
    reviewContent[i]=0
    reviewForExcel[i]=0
    helpfulnessForExcel[i]=0

(reviewForExcel,reviewTitle,reviewContent)=calcuateReviewMarks(npdata,length,12,13,length,reviewTitle,reviewContent)
helpfulnessForExcel=calcuateHelpfulness(npdata,length,8,9,length,0,0)


idPosition=3

# Calcuate the kinds number of data.
count=1
for i in range(0,length-1):
    if str(npdata[i+1,3]).upper()!=str(npdata[i,3]).upper():
        count+=1

idList=np.empty(count+1,dtype=int)
idList[0]=0
cacheOfIdList=0
for i in range(0,length-1):
    if str(npdata[i+1,3]).upper()!=str(npdata[i,3]).upper():
        idList[cacheOfIdList+1]=i+1
        cacheOfIdList+=1

idList[count]=length
everyExcelSize=np.empty(count)

for i in range(0,count):
    temp=0
    everyExcelSize[i]=0
    for j in range(0,idList[i+1]-idList[i]):
        if str(npdata[idList[i+1]-j-1,14])[0:2]!=str(temp):
            temp=str(npdata[idList[i+1]-j-1,14])[0:2]
            everyExcelSize[i]+=1
print(idList)
print(everyExcelSize)
for i in range(0,count):
    temp=0
    outputExcel=np.empty((int(everyExcelSize[i]),22),dtype=float)
    for m in range(0,int(everyExcelSize[i])):
        for n in range(0,22):
            outputExcel[m,n]=0
    order=-1
    for j in range(0,idList[i+1]-idList[i]):
        if str(npdata[idList[i+1]-j-1,14])[0:2]!=str(temp):
            temp=str(npdata[idList[i+1]-j-1,14])[0:2]
            order+=1
            if j!=0:
                outputExcel[order,0]=outputExcel[order-1,0]
                outputExcel[order,2]=outputExcel[order-1,2]
                outputExcel[order,3]=outputExcel[order-1,3]
                outputExcel[order,4]=outputExcel[order-1,4]
                outputExcel[order,5]=outputExcel[order-1,5]
                outputExcel[order,17]=outputExcel[order-1,17]
                outputExcel[order,18]=outputExcel[order-1,18]
                outputExcel[order,19]=outputExcel[order-1,19]
                outputExcel[order,20]=outputExcel[order-1,20]
                outputExcel[order,21]=outputExcel[order-1,21]
        outputExcel[order,0]+=1
        if str(npdata[idList[i+1]-j-1,11])=="Y":
            outputExcel[order, 1]+=1
        outputExcel[order,2]+=npdata[idList[i+1]-j-1,7]
        outputExcel[order,3]+=reviewTitle[idList[i+1]-j-1]
        outputExcel[order,4]+=reviewContent[idList[i+1]-j-1]
        outputExcel[order,5]+=helpfulnessForExcel[idList[i+1]-j-1]
        outputExcel[order,int(npdata[idList[i+1]-j-1,7])+11]+=1
        outputExcel[order,int(npdata[idList[i+1]-j-1,7])+16]+=1
        if str(npdata[idList[i+1]-j-1,10])=="Y":
            outputExcel[order,6]+=1
        outputExcel[order,7]+=npdata[idList[i+1]-j-1,7]
        outputExcel[order,8]+=reviewTitle[idList[i+1]-j-1]
        outputExcel[order,9]+=reviewContent[idList[i+1]-j-1]
        outputExcel[order,10]+=helpfulnessForExcel[idList[i+1]-j-1]
        outputExcel[order,11]+=1
    for k in range(0,int(everyExcelSize[i])):
        outputExcel[k,2]/=outputExcel[k,0]
        outputExcel[k,3]/=outputExcel[k,0]
        outputExcel[k,4]/=outputExcel[k,0]
        outputExcel[k,5]/=outputExcel[k,0]
        outputExcel[k,7]/=outputExcel[k,11]
        outputExcel[k,8]/=outputExcel[k,11]
        outputExcel[k,9]/=outputExcel[k,11]
        outputExcel[k,10]/=outputExcel[k,11]
        # if k>0:
            # outputExcel[k,7]/=(outputExcel[k,0]-outputExcel[k-1,0])
            # outputExcel[k,8]/=(outputExcel[k,0]-outputExcel[k-1,0])
            # outputExcel[k,9]/=(outputExcel[k,0]-outputExcel[k-1,0])
            # outputExcel[k,10]/=(outputExcel[k,0]-outputExcel[k-1,0])
    x=range(0,int(everyExcelSize[i]))

    plt.cla()
    plt.clf()

    plt.title("The "+name+" "+str(npdata[idList[i],3])+" Trend of Stage Star Ratings")
    plt.xlabel("Time Circle (Month)")
    plt.ylabel("Amount")
    plt.plot(x,outputExcel[:,12],label="Amount of 1 Star")
    plt.plot(x,outputExcel[:,13],label="Amount of 2 Star")
    plt.plot(x,outputExcel[:,14],label="Amount of 3 Star")
    plt.plot(x,outputExcel[:,15],label="Amount of 4 Star")
    plt.plot(x,outputExcel[:,16],label="Amount of 5 Star")
    plt.legend(loc="best",framealpha=0.3)
    plt.savefig("../"+name+"/2d/StageStars"+name+"-"+str(i)+"-"+str(npdata[idList[i],3])+".png")

    plt.cla()
    plt.clf()

    plt.title("The "+name+" "+str(npdata[idList[i],3])+" Trend of Star Ratings")
    plt.xlabel("Time Circle (Month)")
    plt.ylabel("Amount")
    plt.plot(x,outputExcel[:,17],label="Amount of 1 Star")
    plt.plot(x,outputExcel[:,18],label="Amount of 2 Star")
    plt.plot(x,outputExcel[:,19],label="Amount of 3 Star")
    plt.plot(x,outputExcel[:,20],label="Amount of 4 Star")
    plt.plot(x,outputExcel[:,21],label="Amount of 5 Star")
    plt.legend(loc="best",framealpha=0.3)
    plt.savefig("../"+name+"/2d/StarsTotal"+name+"-"+str(i)+"-"+str(npdata[idList[i],3])+".png")
    plt.cla()
    plt.clf()

    plt.title("The "+name+" "+str(npdata[idList[i],3])+" Situation")
    plt.xlabel("Time Circle (Month)")
    plt.ylabel("Trend")
    plt.yticks([])
    plt.plot(x,outputExcel[:,2],label="Star Rating")
    plt.plot(x,outputExcel[:,3],label="Review Title Score")
    plt.plot(x,outputExcel[:,4],label="Review Content Score")
    plt.plot(x,outputExcel[:,5],label="Helpfulness")
    plt.legend(loc="best",framealpha=0.3)
    plt.savefig("../"+name+"/2b/Total-"+name+"-"+str(i)+"-"+str(npdata[idList[i],3])+".png")


    plt.cla()
    plt.clf()

    plt.title("The "+name+" "+str(npdata[idList[i],3])+" Situation")
    plt.xlabel("Time Circle (Month)")
    plt.ylabel("Trend")
    plt.yticks([])
    plt.plot(x,outputExcel[:,7],label="Star Rating")
    plt.plot(x,outputExcel[:,8],label="Review Title Score")
    plt.plot(x,outputExcel[:,9],label="Review Content Score")
    plt.plot(x,outputExcel[:,10],label="Helpfulness")
    plt.legend(loc="best",framealpha=0.3)
    plt.savefig("../"+name+"/2b/Period-"+name+"-"+str(i)+"-"+str(npdata[idList[i],3])+".png")

    plt.cla()
    plt.clf()

    plt.title("The "+name+" "+str(npdata[idList[i],3])+" Situation")
    plt.xlabel("Time Circle (Month)")
    plt.ylabel("Trend")
    plt.yticks([])
    plt.plot(x,outputExcel[:,2],label="Star Rating")
    plt.plot(x,outputExcel[:,3],label="Review Title Score")
    plt.plot(x,outputExcel[:,4],label="Review Content Score")
    plt.plot(x,outputExcel[:,5],label="Helpfulness")
    plt.legend(loc="best",framealpha=0.3)
    plt.savefig("../"+name+"/2d/Total"+name+"-"+str(i)+"-"+str(npdata[idList[i],3])+".png")


    plt.cla()
    plt.clf()

    plt.title("The "+name+" "+str(npdata[idList[i],3])+" Situation")
    plt.xlabel("Time Circle (Month)")
    plt.ylabel("Trend")
    plt.yticks([])
    plt.plot(x,outputExcel[:,7],label="Star Rating")
    plt.plot(x,outputExcel[:,8],label="Review Title Score")
    plt.plot(x,outputExcel[:,9],label="Review Content Score")
    plt.plot(x,outputExcel[:,10],label="Helpfulness")
    plt.legend(loc="best",framealpha=0.3)
    plt.savefig("../"+name+"/2d/Period"+name+"-"+str(i)+"-"+str(npdata[idList[i],3])+".png")

    outputExcelDataFrame=pd.DataFrame(outputExcel)
    outputExcelDataFrame.columns = ["Total","Purchased","Star Ratings","Review Title Score","Review Content Score","Helpfulness","Vine","Stage Star Ratings","Stage Title Score","Stage Content Score","Stage Helpfulness","Stage Total","1 Star","2 Star","3 Star","4 Star", "5 Star", "S1","S2","S3","S4","S5"]
    writer=pd.ExcelWriter("../"+name+"/2d/ForStars"+name+"-"+str(i)+"-"+str(npdata[idList[i],3])+'-2d.xlsx')
    outputExcelDataFrame.to_excel(writer, float_format='%.5f')
    writer.save()


print(0)
