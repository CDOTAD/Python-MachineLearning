# -*- coding:utf-8 -*-

#import  numpy.core.multiarray
#import numpy
import pandas
import math
import pickle

"""
朴素的基于用户的协同过滤算法。
用户对于物品只有喜欢和不喜欢。

相似度计算公式为余弦相似度 ：
    
    W_uv = |N(u) and N(v)|/sqrt(|N(u)| * |N(v)|)
    
N(u)表示用户u曾经有过正反馈的物品集合，N(v)表示用户v曾经有过正反馈的物品集合。

根据用户之间的兴趣相似度，给用户推荐和他兴趣最相似的K个用户喜欢的物品中，指定用户最可能感兴趣的K个物品

"""


class NaiveUserCollaborativeFiltering:

    def __init__(self,user,data,k = 10,similar_matrix=None):

        self.user_recommend = user

        self.data_set = data

        self.topK = k

        if type(similar_matrix) == type("str"):

            self.similar_matrix = pickle.load(open(similar_matrix,'rb'))

        else:

            self.similar_matrix = None

        return

    """
    相似度矩阵运算
    
    相似度公式为余弦相似度：
    
        W_uv = |N(u) and N(v)|/sqrt(|N(u)| * |N(v)|)
        
    N(u)表示用户u曾经有过正反馈的物品集合，N(v)表示用户v曾经有过正反馈的物品集合。
    """
    def user_similarity(self,store=False):

        train = self.data_set

        item_users=dict()

        for u,items in train.items():

            for i in items:

                if i not in item_users:
                    item_users[i] = set()

                item_users[i].add(u)

        #print(item_users)

        scale_matrix=dict()

        neighbor_matrix=dict()

        for i,users in item_users.items():

            for u in users:

                if u not in neighbor_matrix:

                    neighbor_matrix[u] = 0

                neighbor_matrix[u] += 1

                for v in users:

                    if u not in scale_matrix:

                        scale_matrix[u] = dict()

                    if v not in scale_matrix[u]:

                        scale_matrix[u][v]=0

                    if u == v:
                        continue

                    scale_matrix[u][v] += 1

        #print(scale_matrix)

        similar_matrix=dict()

        for u,related_users in scale_matrix.items():

            similar_matrix[u]=dict()

            for v, cuv in related_users.items():

                similar_matrix[u][v] = cuv / math.sqrt(neighbor_matrix[u] * neighbor_matrix[v])

        if store:

            store_file=open('user_similar_matrix_large','wb')

            pickle.dump(similar_matrix,store_file)

        return similar_matrix

    def recommend(self):

        train = self.data_set

        if self.similar_matrix is None:

            self.similar_matrix = self.user_similarity()

        rank = dict()

        interacted_item = self.data_set[self.user_recommend]

        user_similarity=self.similar_matrix[self.user_recommend]

        for v,wuv in sorted(user_similarity.items(),key=lambda user_similarity:user_similarity[1],reverse=True)[0:self.topK]:

            for i in train[v]:

                if i in interacted_item:
                    continue

                if i not in rank:

                    rank[i]=0

                rank[i] += wuv

            rank = sorted(rank.items(),key=lambda rank:rank[1],reverse=True)[0:self.topK]

            return rank


class IUFUserCollaborativeFiltering(NaiveUserCollaborativeFiltering):

    """
    相似度矩阵运算
    
    相似度公式为：
    
        W_uv = SUM(1 / log(1 + |N(i)|)/sqrt(|N(u)| * |N(v)|)  (i∈N(u) and N(v))
        
    N(u)表示用户u曾经有过正反馈的物品集合，N(v)表示用户v曾经有过正反馈的物品集合。
    """
    def user_similarity(self):

        train = self.data_set

        item_users = dict()

        for u, items in train.items():

            for i in items:

                if i not in item_users:
                    item_users[i] = set()

                item_users[i].add(u)

        scale_matrix = dict()

        neighbor_matrix = dict()

        for i, users in item_users.items():

            for u in users:

                if u not in neighbor_matrix:
                    neighbor_matrix[u] = 0

                neighbor_matrix[u] += 1

                for v in users:

                    if u not in scale_matrix:

                        scale_matrix[u] = dict()

                    if v not in scale_matrix[u]:
                        scale_matrix[u][v] = 0

                    if u == v:
                        continue

                    scale_matrix[u][v] += 1 / math.log(1 + len(users))

        similar_matrix = dict()

        for u, related_users in scale_matrix.items():

            similar_matrix[u] = dict()

            for v, cuv in related_users.items():
                similar_matrix[u][v] = cuv / math.sqrt(neighbor_matrix[u] * neighbor_matrix[v])

        return similar_matrix

dataSet=pandas.read_csv('./src/ratings.csv');


myData=dict()

for i in range(len(dataSet.userId)):

    userId=dataSet.userId[i]

    if userId not in myData:
        myData[userId]=set()

    myData[userId].add(dataSet.movieId[i])

print myData
print(len(myData))

navieUCF=NaiveUserCollaborativeFiltering(user='1',data=myData,k=3)

sMatrix=navieUCF.user_similarity(store=True)

print(sMatrix)

aData=dict()

aData['A']=set(['a','b','d'])
aData['B']=set(['a','c'])
aData['C']=set(['b','e'])
aData['D']=set(['c','d','e'])

#navieUCF=NaiveUserCollaborativeFiltering(user='A',data=aData,k=3,similar_matrix="user_similar_matrix")

#print(navieUCF.similar_matrix)

#sMatrix = navieUCF.user_similarity(store=False)

#print(sMatrix)

#iufUCF=IUFUserCollaborativeFiltering(user='A',data=aData,k=3)

#sMatrixIUF = iufUCF.user_similarity()

#print(sMatrixIUF)

#print (1)

#def naive_user_similar(user,k=10,similar_matrix=None,store=False):

    #navieU=NaiveUserCollaborativeFiltering(user=user,data=aData,k=k,similar_matrix=similar_matrix)

    #return navieU.user_similarity(store=store)

#def naive_recommend(user,k):
    #navieU = NaiveUserCollaborativeFiltering(user=user,k=k,)

#def add(num1,num2):

    #return num1 + num2