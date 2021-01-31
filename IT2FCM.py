import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist

df = pd.read_excel('veri3.xlsx',header=None)
df=df.to_numpy()
data = df[:]
#data=data / np.linalg.norm(data)
cluster=10
m1=2.5
m2=4.5
m=3
def DistanceMatrix(center,x,distance):
    distanceMatrix = cdist(x, center, distance).T
    return distanceMatrix
def MembershipMatrix1(distanceMatrix,r):
    c, n = distanceMatrix.shape
    membershipMatrix = np.zeros((c, n))
    p = 2 / (r - 1)
    [row2, col2] = np.where((np.isnan(distanceMatrix)))
    zs2 = np.size(row2)
    for j in range(zs2):
        distanceMatrix[row2(j), col2(j)] = 1
    [row, col] = np.where(distanceMatrix == 0)
    dp = 1 / (distanceMatrix** p)
    dsum = np.sum(dp, axis=0)
    for j in range(n):
        membershipMatrix[:, j]=dp[:, j]/ dsum[j]
    zs = np.size(row)
    for j in range(zs):
        membershipMatrix[:, col(j)] = np.zeros(c, 1)
        membershipMatrix[row(j), col(j)] = 1
    return membershipMatrix
def MembershipMatrix(center,x,distance,r):
    distanceMatrix=DistanceMatrix(center,x,distance)
    membershipMatrix = MembershipMatrix1(distanceMatrix, r)
    return membershipMatrix
def GetCenter(x,u,r):
    try:
        n= x.shape[0]
    except IndexError:
        n=1
    try:
        m = x.shape[1]
    except IndexError:
        m = 1
    #shape = (x.shape[0], x.shape[1])
    #n,m=np.array([x]).shape
    ur=u**r
    center = np.dot(ur,x)/(np.dot(ur,np.ones((n,m))))
    return center
def KMCL(x,ulower,uupper,r):
    e=1
    l=1
    L=20
    dlist=np.zeros((L,1))
    n,m=x.shape
    u=(ulower+uupper)/2
    c,tt=u.shape
    v1=GetCenter(x,u,r)
    v2=np.copy(v1)
    while(l<=L):
        for i in range(m):
            for j in range(c):
                for k in range(n):
                    if x[k,i]<=v1[j,i]:
                        u[j,k]=uupper[j,k]
                    else:
                        u[j,k]=ulower[j,k]
            v2[:,[i]]=GetCenter(x[:,[i]],u,r)
        d=np.sum(np.sum(np.abs(v1-v2),axis=0),axis=0)
        dlist[l]=d
        if d<e:
            v=np.copy(v2)
            break
        else:
            v1=np.copy(v2)
    return v,dlist
def KMCR(x,ulower,uupper,r):
    e=1
    l=1
    L=20
    dlist=np.zeros((L,1))
    n,m=x.shape
    u=(ulower+uupper)/2
    c,tt=u.shape
    v1=GetCenter(x,u,r)
    v2=np.copy(v1)
    while(l<=L):
        for i in range(m):
            for j in range(c):
                for k in range(n):
                    if x[k,i]>=v1[j,i]:
                        u[j,k]=uupper[j,k]
                    else:
                        u[j,k]=ulower[j,k]
            v2[:,[i]]=GetCenter(x[:,[i]],u,r)
        d=np.sum(np.sum(np.abs(v1-v2)))
        dlist[l]=d
        if d<e:
            v=np.copy(v2)
            break
        else:
            v1=np.copy(v2)
    return v,dlist
def HardPartition(center,x,dis):
    n,kk=x.shape
    result=np.zeros((n,1))
    d=DistanceMatrix(center,x,dis)
    mind=d.min(axis=0)
    for i in range(n):
        result[i] =(d[:,[i]]==mind[i]).argmax()
        #result[i]=np.where(d[:,i]==mind(i),1)
    return result
def IT2FCM(x,vs,dis,r,r1,r2):
    v1=vs
    e=10**(-5)
    l = 0
    L = 20
    dlist=np.zeros((L+1,1))
    c=np.size(vs,l)
    while (l <= L):
        u1=MembershipMatrix(v1,x,dis,r1)
        u2 = MembershipMatrix(v1, x, dis, r2)
        ulower=np.minimum(u1,u2)
        uupper=np.maximum(u1,u2)
        vl = KMCL(x, ulower, uupper, r)[0]
        vr = KMCR(x, ulower, uupper, r)[0]
        v2 = (vl + vr) / 2
        d = np.sum(np.sum(np.abs(v1 - v2)))
        if d>e:
            dlist[l]=d
            v1=np.copy(v2)
            l=l+1
        else:
            break
    ve=np.copy(v2)

    ve=v2[v2[:, 0].argsort(),]
    result=HardPartition(ve[:],x,dis)
    return result,ve,vl,vr,dlist
def merge(a,b):
    x = np.squeeze(np.stack((a, b), axis=1))
    #x = x[x[:, 1].argsort()]
    return x
def member(a):
    return merge(MembershipMatrix(vl,[a],"euclidean",m),MembershipMatrix(vr,[a],"euclidean",m))


np.set_printoptions(suppress=True)
t2t,t2v,vl,vr,dlist=IT2FCM(data,np.random.rand(cluster,1)*12,'euclidean',m,m1,m2)

print(merge(vl,vr))

#print(getdist([5]))
#print(vl)
#print(member([5.0]))
#print(vr)
#print(DistanceMatrix(vl,[[5]],"euclidean"))
#print(MembershipMatrix(5,vl,"euclidean"))
#print(MembershipMatrix(5,vr,"euclidean"))
