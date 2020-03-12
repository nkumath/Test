# Code for HairDryer

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
name="hair_dryer"

path="C:\\Users\\Eternally Ascend\\Desktop\\Problem_C_Data\\"+name+".tsv"

# Read the data.
data=pd.read_csv(path, sep='\t', header=0)
npdata=np.array(data)
length=npdata.shape[0]

# The scord of Star Rating.
star=np.array(npdata[:,7])

stars=np.empty(5)
vine=np.empty(5)
purchase=np.empty(5)

for i in range(0,5):
    stars[i]=0
    vine[i]=0
    purchase[i]=0

for i in npdata[:,7]:
    stars[int(i)-1]+=1

for i in range(0,length):
    if npdata[i,10] == "Y":# or npdata[i:10] == "y":
        vine[int(star[i])-1]+=1

for i in range(0,length):
    if npdata[i,11] == "Y":# or npdata[i,11] == "y":
        purchase[int(star[i])-1]+=1

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

rateAlpha=alpha(rating,mini,middle,maxn,rateAlpha,length)

# The record of reviews.
titles=np.array(npdata[:,12])
contents=np.array(npdata[:,13])

titlePositive=np.empty(length,dtype=float)
contentPositive=np.empty(length,dtype=float)

titleSubjectivity=np.empty(length,dtype=float)
contentSubjectivity=np.empty(length,dtype=float)

for i in range(0,length):
    temp=textblob.TextBlob(str(titles[i]))
    titlePositive[i]=temp.sentiment.polarity
    titleSubjectivity[i]=temp.sentiment.subjectivity
    temp=textblob.TextBlob(str(contents[i]))
    contentPositive[i]=temp.sentiment.polarity
    contentSubjectivity[i]=temp.sentiment.subjectivity

# Function for calcuating the score of reviews.
def markReviews(pos,sub,score,length,x,y):
    for i in range(0,length):
        if pos[i]>=0:
            score[i]=pow(x,pos[i])/(sub[i]+y)
        else:
            score[i]=pos[i]*pow(x,pos[i])/(abs(pos[i])*y*(sub[i]+1))

titleScore=np.empty(length,dtype=float)
reviewScore=np.empty(length,dtype=float)
theReviewScore=np.empty(length,dtype=float)

markReviews(titlePositive,titleSubjectivity,titleScore,length,3,0.5)
markReviews(contentPositive,contentSubjectivity,reviewScore,length,3,0.5)

for i in range(0,length):
    theReviewScore[i]=titleScore[i]*0.2+reviewScore[i]*0.8

# Figures.

plt.cla()
plt.clf()

# Star Rating.
plt.title("Star Rating of "+name)
plt.xlabel("Star Rating")
plt.ylabel("Amount")
plt.xticks(range(1,6))
plt.bar(range(0,5),stars,label="Total")
plt.bar(range(0,5),purchase,label="Purchased")
plt.bar(range(0,5),vine,label="Vine")
plt.legend()
plt.savefig("./"+name+"/"+name+"Star.png")

plt.cla()
plt.clf()

# Review score.
reviewMin=0
reviewMax=0

for i in reviewScore:
    if float(i)>reviewMax:
        reviewMax=math.ceil(float(i))
    if float(i)<reviewMin:
        reviewMin=math.floor(float(i))

reviewMap=np.empty(reviewMax-reviewMin+1)

for i in range(0,reviewMax-reviewMin+1):
    reviewMap[i]=0

for i in theReviewScore:
    reviewMap[int((round(i))-reviewMin)]+=1

plt.title("Review Score of "+name)
plt.xlabel("Score")
plt.ylabel("Amount")
plt.xticks(range(0,reviewMax-reviewMin))
plt.bar(range(0,reviewMax-reviewMin),reviewMap[0:reviewMax-reviewMin],label="Reviews Score")
plt.legend()
plt.savefig("./"+name+"/"+name+"ReviewScore.png")

plt.cla()
plt.clf()

# Usefulness Rating.
usefulGroup=np.empty(101)
for i in range(0,101):
    usefulGroup[i]=0

for i in rateAlpha:
    cache=int(math.floor(100.0*i))
    if cache>=0:
        usefulGroup[cache]+=1

plt.title("Helpful Votes of "+name)
plt.xlabel("Helpfulness (Percent)")
# plt.xlabel("The Rate of Helpful Votes/Total Votes (Percent)")
plt.ylabel("Amount")
plt.plot(range(0,100),usefulGroup[1:],label="Helpful Votes")
plt.legend()
plt.savefig("./"+name+"/"+name+"Usefulness.png")




plt.cla()
plt.clf()

# Star and reviews.
levelOfStar=1
reviewGroup=np.empty(reviewMax-reviewMin+1)

for i in range(0,reviewMax-reviewMin+1):
    reviewGroup[i]=0

for i in range(0,length):
    if star[i]==levelOfStar:
        reviewGroup[math.ceil(theReviewScore[i])+1]+=1

plt.title("The "+str(levelOfStar)+" Star Rating and Reviews of "+name)
plt.xlabel("Reviews Score")
plt.ylabel("Amount")
plt.plot(range(0,reviewMax-reviewMin),reviewGroup[0:reviewMax-reviewMin],label="Amount")
plt.legend()
plt.savefig("./"+name+"/"+name+"Star "+str(levelOfStar)+" Reviews.png")

plt.cla()
plt.clf()

levelOfStar=2
reviewGroup=np.empty(reviewMax-reviewMin+1)

for i in range(0,reviewMax-reviewMin+1):
    reviewGroup[i]=0

for i in range(0,length):
    if star[i]==levelOfStar:
        reviewGroup[math.ceil(theReviewScore[i])+1]+=1

plt.title("The "+str(levelOfStar)+" Star Rating and Reviews of "+name)
plt.xlabel("Reviews Score")
plt.ylabel("Amount")
plt.plot(range(0,reviewMax-reviewMin),reviewGroup[0:reviewMax-reviewMin],label="Amount")
plt.legend()
plt.savefig("./"+name+"/"+name+"Star "+str(levelOfStar)+" Reviews.png")

plt.cla()
plt.clf()

levelOfStar=3
reviewGroup=np.empty(reviewMax-reviewMin+1)

for i in range(0,reviewMax-reviewMin+1):
    reviewGroup[i]=0

for i in range(0,length):
    if star[i]==levelOfStar:
        reviewGroup[math.ceil(theReviewScore[i])+1]+=1

plt.title("The "+str(levelOfStar)+" Star Rating and Reviews of "+name)
plt.xlabel("Reviews Score")
plt.ylabel("Amount")
plt.plot(range(0,reviewMax-reviewMin),reviewGroup[0:reviewMax-reviewMin],label="Amount")
plt.legend()
plt.savefig("./"+name+"/"+name+"Star "+str(levelOfStar)+" Reviews.png")

plt.cla()
plt.clf()

levelOfStar=4
reviewGroup=np.empty(reviewMax-reviewMin+1)

for i in range(0,reviewMax-reviewMin+1):
    reviewGroup[i]=0

for i in range(0,length):
    if star[i]==levelOfStar:
        reviewGroup[math.ceil(theReviewScore[i])+1]+=1

plt.title("The "+str(levelOfStar)+" Star Rating and Reviews of "+name)
plt.xlabel("Reviews Score")
plt.ylabel("Amount")
plt.plot(range(0,reviewMax-reviewMin),reviewGroup[0:reviewMax-reviewMin],label="Amount")
plt.legend()
plt.savefig("./"+name+"/"+name+"Star "+str(levelOfStar)+" Reviews.png")

plt.cla()
plt.clf()

levelOfStar=5
reviewGroup=np.empty(reviewMax-reviewMin+1)

for i in range(0,reviewMax-reviewMin+1):
    reviewGroup[i]=0

for i in range(0,length):
    if star[i]==levelOfStar:
        reviewGroup[math.ceil(theReviewScore[i])+1]+=1

plt.title("The "+str(levelOfStar)+" Star Rating and Reviews of "+name)
plt.xlabel("Reviews Score")
plt.ylabel("Amount")
plt.plot(range(0,reviewMax-reviewMin),reviewGroup[0:reviewMax-reviewMin],label="Amount")
plt.legend()
plt.savefig("./"+name+"/"+name+"Star "+str(levelOfStar)+" Reviews.png")

plt.cla()
plt.clf()

plt.title("Star and Reviews of "+name)
plt.xlabel("Star")
plt.xticks(range(1,6))
plt.ylabel("Reviews Score")
theAnsRS=np.empty(length)
for i in range(0,length):
    theAnsRS[i]=0
    theAnsRS[i]=theReviewScore[i]-reviewMin
plt.scatter(star,theAnsRS,label="Remark Score",s=1)
plt.legend()
plt.savefig("./"+name+"/"+name+"StarAndReview.png")

plt.cla()
plt.clf()

# Star and usefulness.
levelOfStar=1
starAndUsefulness=np.empty(101)

for i in range(0,101):
    starAndUsefulness[i]=0

for i in range(0,length):
    if star[i]==levelOfStar:
        starAndUsefulness[math.floor(100*rateAlpha[i])]+=1

plt.title("The "+str(levelOfStar)+" Star Rating and Usefulness of "+name)
plt.xlabel("Usefulness (Percent)")
plt.ylabel("Amount")
plt.plot(range(0,101),starAndUsefulness,label="Votes")
plt.legend()
plt.savefig("./"+name+"/"+name+"Star "+str(levelOfStar)+" Usefulness.png")

plt.cla()
plt.clf()

levelOfStar=2
starAndUsefulness=np.empty(101)

for i in range(0,101):
    starAndUsefulness[i]=0

for i in range(0,length):
    if star[i]==levelOfStar:
        starAndUsefulness[math.floor(100*rateAlpha[i])]+=1

plt.title("The "+str(levelOfStar)+" Star Rating and Usefulness of "+name)
plt.xlabel("Usefulness (Percent)")
plt.ylabel("Amount")
plt.plot(range(0,101),starAndUsefulness,label="Votes")
plt.legend()
plt.savefig("./"+name+"/"+name+"Star "+str(levelOfStar)+" Usefulness.png")

plt.cla()
plt.clf()

levelOfStar=3
starAndUsefulness=np.empty(101)

for i in range(0,101):
    starAndUsefulness[i]=0

for i in range(0,length):
    if star[i]==levelOfStar:
        starAndUsefulness[math.floor(100*rateAlpha[i])]+=1

plt.title("The "+str(levelOfStar)+" Star Rating and Usefulness of "+name)
plt.xlabel("Usefulness (Percent)")
plt.ylabel("Amount")
plt.plot(range(0,101),starAndUsefulness,label="Votes")
plt.legend()
plt.savefig("./"+name+"/"+name+"Star "+str(levelOfStar)+" Usefulness.png")

plt.cla()
plt.clf()

levelOfStar=4
starAndUsefulness=np.empty(101)

for i in range(0,101):
    starAndUsefulness[i]=0

for i in range(0,length):
    if star[i]==levelOfStar:
        starAndUsefulness[math.floor(100*rateAlpha[i])]+=1

plt.title("The "+str(levelOfStar)+" Star Rating and Usefulness of "+name)
plt.xlabel("Usefulness (Percent)")
plt.ylabel("Amount")
plt.plot(range(0,101),starAndUsefulness,label="Votes")
plt.legend()
plt.savefig("./"+name+"/"+name+"Star "+str(levelOfStar)+" Usefulness.png")

plt.cla()
plt.clf()

levelOfStar=5
starAndUsefulness=np.empty(101)

for i in range(0,101):
    starAndUsefulness[i]=0

for i in range(0,length):
    if star[i]==levelOfStar:
        starAndUsefulness[math.floor(100*rateAlpha[i])]+=1

plt.title("The "+str(levelOfStar)+" Star Rating and Usefulness of "+name)
plt.xlabel("Usefulness (Percent)")
plt.ylabel("Amount")
plt.plot(range(0,101),starAndUsefulness,label="Votes")
plt.legend()
plt.savefig("./"+name+"/"+name+"Star "+str(levelOfStar)+" Usefulness.png")

plt.cla()
plt.clf()

plt.title("Star and Usefulness of "+name)
plt.xlabel("Star")
plt.xticks(range(1,6))
plt.ylabel("Usefulness (Percent)")
plt.scatter(star,rateAlpha,label="Usefulness",s=1)
plt.savefig("./"+name+"/"+name+"StarAndUsefulness.png")

plt.cla()
plt.clf()

# Reviews and usefulness.
plt.title("Helpfulness and Reviews of "+name)
plt.xlabel("Helpfulness (Percent)")
plt.ylabel("Review Score")
plt.yticks(range(0,reviewMax-reviewMin))
plt.scatter(rateAlpha,theReviewScore,label="Reviews Score",s=1)
plt.savefig("./"+name+"/"+name+"HelpfulnessAndReviews.png")

plt.cla()
plt.clf()