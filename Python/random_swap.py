"""
Author: Jiawei Yang
Last updated date: 23th May, 2018
V 1.1

log: fixed bug (in v 1.0) pointed out by Pasi on 22th May 2018
refer:http://cs.uef.fi/pages/franti/research/rs.txt
"""
import random
import numpy as np
from numpy import genfromtxt
import copy
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import timeit
from scipy.spatial import ConvexHull, distance
import collections


############### start random swap algorithm ###############

def PerformRS(X,iterationsRS,iterationKmean,clusters):
    
    """
    ----------
    Performs Random Swap -algorithm for given parameters.

    Uses the k_means function implemented in the file.

    Parameters:
    ----------
    X : N*V dimensional array with N datapoints
        The actual coordinates of the datapoints

    iterationsRS : int
        Stops random swap after the amount of iterations

    clusters : int
        Initializes random_swap with given amount of clusters

    iterationKmean : int
        Stops k-means after the amount of  iterations
    ----------

    Output:
    ----------
    centroids : V dimensional array with C datapoints
        Predefined coordinates of centroids

    partition : scalar array with N datapoints
        Information about which datapoint belongs to which centroid
    ----------
    ----------

    """
    
    
    #/* initial solution */
    # option 1.  select points ramdomly
    C = SelectRandomRepresentatives(X,clusters)
    P = OptimalPartition(C,X)
    
    #soption 2. elect points from centroid by k-means
    # kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
    # P=kmeans.labels_
    # C=kmeans.cluster_centers_

    err=ObjectiveFunction(P,C,X)
    print("Intinal MSE:",err)
    it=0
    while it <iterationsRS:
        C_new,j= RandomSwap(copy.deepcopy(C),X)
        P_new= LocalRepartition(copy.deepcopy(P),C_new,X,j)
        P_new,C_new= K_means(P_new,C_new,X,iterationKmean)
        new_err=ObjectiveFunction(P_new,C_new,X)
        if  new_err<err :
           P=copy.deepcopy(P_new)
           C=copy.deepcopy(C_new)
           print("Iteration:",it,"MSE=",new_err)
           err=new_err
        it+=1
    #print("MSE:",ObjectiveFunction(P,C,X))
    return P,C



def K_means(P,C,X,T):
    #/* performs two K-means iterations */
    for i in range(T):
        #/* OptimalRepresentatives-operation should be before
        #OptimalPartition-operation, because we have previously tuned
        #partition with LocalRepartition-operation */
        C = OptimalRepresentatives(P,X,len(C))
        P = OptimalPartition(C,X)

    return P,C



def OptimalPartition(C,X):
    N=len(X)
    P=[0]*N
    for i in range(N):
        P[i] = FindNearestRepresentative(C,X[i],len(C))
    return P



def OptimalRepresentatives(P,X,clusters):
    #/* initialize Sum[1..k] and Count[1..k] by zero values! */

    #/* sum vector and count for each partition */
    N=len(X)
    Sum=[0]*N
    Count=[0]*N
    K=clusters
    for i in range(N):
        j = P[i]
        Sum[j] = Sum[j] + X[i]
        Count[j] = Count[j] + 1
    #/* optimal representatives are average vectors */
    C=[[0]]*K
    for i in range(K):
        if Count[i] > 0 :
           C[i] = Sum[i] / Count[i]
    return C



def FindNearestRepresentative(C,x,clusters):
    K=clusters
    j = 0
    for i in range(1,K):
        if Dist(x,C[i]) < Dist(x,C[j]):

           j = i
    return j


def SelectRandomDataObject(C,X,m):
    N=len(X)
    ok = False
    while(not ok):
        i = Random(0,N)
        ok = True
        #/* eliminate duplicates */
        for j in range(m):
            if np.array_equal (C[j],X[i]):
                ok = False
    return X[i]

def SelectRandomRepresentatives(X,clusters):
    C=[[0,0]]*clusters
    for i in range (clusters):
        C[i] = SelectRandomDataObject(C,X,i);
    return C

def RandomSwap(C,X):
    j = Random(0,len(C))
    C[j] = SelectRandomDataObject(C,X,clusters)
    return C,j


def LocalRepartition(P,C,X,j):
    #k=clusters
    N=len(X)
    #/* object rejection */
    for i in range(N):
        if P[i] ==j:
            P[i] = FindNearestRepresentative(C,X[i],len(C))
    #/* object attraction */
    for i in range(N):
        if Dist(X[i],C[j]) < Dist(X[i],C[P[i]]):
            P[i] = j
    return P



#/* this (example) objective function is sum of squared distances
#   of the data object to their cluster representatives */

def ObjectiveFunction(P,C,X):

   #(MSE=TSE/(N*V)

    sum = 0
    N=len(X)
    for i in range(N):
        sum = sum +np.sum((X[i]-C[P[i]])**2)#Calculates total squared error (TSE)
    return sum/(N*len(X[0])) #calculates nMSE =(TSE/(N*V))



def Dist(x1,x2):     #calculates euclidean distance between vectors x1 and x2
    return np.sqrt(np.sum((x1-x2)**2))
def Random(a,b):     #returns random number between a..b
    re=random.randint(a,b-1)
    return re

############### end random swap algorithm ###############

###############
#/* useful methods */
def load_points(filename):
    data = genfromtxt(filename)
    return data


def plotXY(data,data2):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        r = 1 + 99 * np.random.random((100, 2))
        b = np.random.random((100, 2))
        x1, y1 = np.array(data).T
        plt.plot(x1, y1,'.',markersize=5,color='0.5')
        x2, y2 = np.array(data2).T
        plt.plot(x2, y2,'.',markersize=15,color='r')
        plt.legend(loc='upper left')
        plt.axis('off')
        plt.show()

def cal_Centroid_Index(cluster_centers,cluster_centers_pred):
    cost = distance.cdist(cluster_centers, cluster_centers_pred, 'euclidean')# Manhattan distance,  'euclidean','minkowski'
    a=np.argmin(cost,axis=0)
    counter=collections.Counter(a)
    zero_count=0
    for i in range(len(cluster_centers)):
        if counter.get(i)==0:
           zero_count +=1
    sum=0
    for i in range(2,len(cluster_centers)):
        if i in counter.keys():
            if counter.get(i)>1:
                sum +=counter.get(i)-1
    if sum>zero_count:
        return sum
    else:
        return zero_count
###############


########### Setting as  below ###########


filename="data/s1.txt"
X=load_points(filename)
X=X[:,[0,1]] # take 2-d points
iterationsRS=10
iterationKmean=2 # perform k-mean step, default is 2
clusters=15 

now=timeit.default_timer()
P,C=PerformRS(X,iterationsRS,iterationKmean,clusters)
now2=timeit.default_timer()

MSE=ObjectiveFunction(P,C,X) #calculate MSE
#print("MSE:",MSE)

###calculate Centroid Index
C_pred=C
real_C_file="data/s1-cb.txt"
C_real=load_points(real_C_file)
CI=cal_Centroid_Index(C_pred,C_real)
#print("Centroid Index:",CI)

print("Total iterations:",iterationsRS,"||Final MSE=",MSE,"||CI=",CI,"||Time=",now2-now)

###plot clusters and data
plotXY(X,C)


# # k-means
# kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
# P=kmeans.labels_
# C=kmeans.cluster_centers_
# ###plot clusters and data
# plotXY(X,C)
# ###calculate MSE
# print(ObjectiveFunction(P,C,X))
# CI=cal_Centroid_Index(C,C_real)
# print("Centroid Index:",CI)
