# -*- coding: UTF-8 -*-

## python3 code

###这里用到了一些数据固化的处理 比如 group_rel_coor=pickle.load(open('group_rel_coor','rb'))
###方便两次做作业的时候不用再重新重csv中清洗数据

import random
import pandas
import pickle
import math
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing


#角度转弧度
def angle2randian(angle):
    return math.pi*angle/180 

#误差计算 为球面两点间距离 将地球看做一个半径为6400km的球体 
def distance_sphere(Lat1,Lon1,Lat2,Lon2):
    R=6400000
    Lat1=angle2randian(Lat1)
    Lon1=angle2randian(Lon1)
    Lat2=angle2randian(Lat2)
    Lon2=angle2randian(Lon2)

    cos = math.cos(Lat1)*math.cos(Lat2)*math.cos(Lon1-Lon2)+math.sin(Lat1)*math.sin(Lat2)

    if cos>1:
        cos=1
    elif cos<-1:
        cos=-1

    sita=math.acos(cos)



    return R*sita


gongcan_2g=pandas.read_csv('hw4/2G/final_2g_gongcan.csv')
test2G=pandas.read_csv('hw4/2G/final_2g_te.csv')
train2G=pandas.read_csv('hw4/2G/final_2g_tr.csv')


#用于匹配基站信息的(LAC,CI)对
lac_list=list(gongcan_2g.LAC)
ci_list=list(gongcan_2g.CI)

tupleList=[(item1,item2) for item1,item2 in zip(lac_list,ci_list)]



#数据分组
group=dict()

for i in range(len(test2G.IMSI)):
    tupleIndex = tupleList.index((test2G.RNCID_1[i],test2G.CellID_1[i]))
    if tupleIndex not in group.keys():
        group[tupleIndex] = []
    print(i)
    group[tupleIndex].append(list(test2G.ix[i]))

for i in range(len(train2G.IMSI)):
    tupleIndex=tupleList.index((train2G.RNCID_1[i],train2G.CellID_1[i]))
    if tupleIndex not in group.keys():
        group[tupleIndex]=[]
    group[tupleIndex].append(list(train2G.ix[i]))




#csv文件列明对应index字典，方便在list中取相应的数据
columns=dict()

for i in range(len(test2G.columns)):
    columns[test2G.columns[i]]=i



#相对GPS
for key ,re in group.items():

    for record in re:

        longitude = record[columns['Longitude']]
        latitude = record[columns['Latitude']]


        rel_longitued = longitude - gongcan_2g.经度[key]
        rel_latitude = latitude - gongcan_2g.纬度[key]

        record[columns['Longitude']] = rel_longitued
        record[columns['Latitude']] = rel_latitude


index_list=list(range(3,columns['Longitude']))
index_list.extend(list(range(columns['Latitude']+1,len(columns.keys())-2)))


# 每个基站组建立一个回归森林
forest_dic=dict()

for key,record in group.items():

    random_forest = RandomForestRegressor(max_depth=30,random_state=2)

    forest_dic[key] = random_forest




error_group=dict()

group_rel_coor=pickle.load(open('group_rel_coor','rb'))

#对于每个分组 随机打乱 取前80%做训练集 后20%做测试集 重复10次 记录误差
for key,record in group_rel_coor.items():
     error_group[key]=[]
     random_forest=forest_dic[key]
     for i in range(10):
         random.shuffle(record)
         trainX=[[re[i] for i in index_list] for re in record]
         trainY=[[re[columns['Longitude']],re[columns['Latitude']]] for re in record]
         random_forest.fit(trainX[:int(len(trainX)*0.8)],trainY[:int(len(trainY)*0.8)])
         testX=trainX[int(len(trainX)*0.8):]
         testY=trainY[int(len(trainY)*0.8):]
         label=random_forest.predict(testX)
         sum = 0
         for i in range(len(label)):
            sum += distance_sphere(testY[i][1]+gongcan_2g.纬度[key],testY[i][0]+gongcan_2g.经度[key],label[i][1]+gongcan_2g.纬度[key],label[i][0]+gongcan_2g.经度[key])
         error_group[key].append(sum / len(label))
         print(sum/len(label))




#其他基站模型预测 某一个基站坐标
other_group_distance_error=dict()

for key in group_rel_coor.keys():
    
    record = group_rel_coor[key]

    error_list=dict()

    base_distance=dict()

    for key_other,random_forest in forest_dic.items():

        base_distance[key_other]=distance_sphere(gongcan_2g.经度[key],gongcan_2g.纬度[key],gongcan_2g.经度[key_other],gongcan_2g.纬度[key_other])

        trainX=[[re[i] for i in index_list] for re in record]
        trainY=[[re[columns['Longitude']],re[columns['Latitude']]]for re in record]

        label=random_forest.predict(trainX)
        
        sum = 0
        
        for i in range(len(label)):

            sum += distance_sphere(trainY[i][1]+gongcan_2g.纬度[key],trainY[i][0]+gongcan_2g.经度[key],label[i][1]+gongcan_2g.纬度[key],label[i][0]+gongcan_2g.经度[key])

        error_list[key_other]=sum/len(label)

    distance_sort=sorted(base_distance.items(),key=lambda base_distance:base_distance[1])
        
    y_list=[]
    for item in distance_sort:
        y_list.append(error_list[item[0]])

    other_group_distance_error[key]=y_list



#基站间距离排序计算 key = 基站编号 value = 按照与该基站距离从近到预远排序
distance_matrix=dict()
for key in keys:

    base_distance=dict()

    for key_other in keys:

        base_distance[key_other]=distance_sphere(gongcan_2g.纬度[key],gongcan_2g.经度[key],gongcan_2g.纬度[key_other],gongcan_2g.经度[key_other])

    distance_sort=sorted(base_distance.items(),key=lambda base_distance:base_distance[1])

    distance_list=[]
    for item in distance_sort:
        distance_list.append(item[0])

    distance_matrix[key]=distance_list
    print(key)



#基于工参表信息基站特征向量建模
def base_station_info_process(base_station_info):
    return_list=[]
    for item in base_station_info:
        if type(item)!=np.float64:
            if len(item)>1:
                item=item[:-1]
            else:
                if item=='是':
                    item =1
                elif item=='否':
                    item = 0
        elif np.isnan(item):
            item = 0
        return_list.append(item)
    return return_list

gongcan_index=list(range(15,23))
gongcan_index.extend(list(range(24,32)))
gongcan_index.extend(list(range(34,36)))
gongcan_index.extend(list(range(38,43)))
gongcan_index.extend(list(range(46,48)))


#基站特征矩阵
baseStation_feature=dict()
for key in keys:
    base_station_info=list(gongcan_2g.ix[key])
    base_station_feature=[base_station_info[i] for i in gongcan_index]
    base_station_feature=base_station_info_process(base_station_feature)

    baseStation_feature[key]=base_station_feature
#基站特征min-max标准化
X=np.array([item for key,item in baseStation_feature.items()])
min_max_scaler=preprocessing.MinMaxScaler()

X_train_minmax=min_max_scaler.fit_transform(X)

for key in keys:
    baseStation_feature[key]=X_train_minmax[keys.index(key)]



#基站相似度矩阵 N * N
base_station_similar_matrix=dict()
for key,vector in baseStation_feature.items():
    sim_dic=dict()
    for key_other,vector_other in baseStation_feature.items():
        sim = vector_other.dot(vector) / (np.sqrt(vector_other.dot(vector_other)) * (np.sqrt(vector.dot(vector))))
        sim_dic[key_other]=sim
    base_station_similar_matrix[key]=sim_dic



#用于统计
other_group_sim_error=dict()
for key in keys:
    
    record = group_rel_coor[key]

    error_list=dict()

    for key_other,random_forest in forest_dic.items():

        
        trainX=[[re[i] for i in index_list] for re in record]
        trainY=[[re[columns['Longitude']],re[columns['Latitude']]]for re in record]

        label=random_forest.predict(trainX)
        
        sum = 0
        
        for i in range(len(label)):

            sum += distance_sphere(trainY[i][1]+gongcan_2g.纬度[key],trainY[i][0]+gongcan_2g.经度[key],label[i][1]+gongcan_2g.纬度[key],label[i][0]+gongcan_2g.经度[key])

        error_list[key_other]=sum/len(label)

    sim_matrix=base_station_similar_matrix[key]
    sim_sort=sorted(sim_matrix.items(),key=lambda sim_matrix:sim_matrix[1],reverse=True)
        
    y_list=[]
    for item in sim_sort:
        y_list.append(error_list[item[0]])

    other_group_sim_error[key]=y_list




#基站周围道路分布建模
group_distribute_matrix=dict()
for key,record in group_rel_coor.items():
    gongcan_lat=gongcan_2g.经度[key]
    gongcan_lon=gongcan_2g.纬度[key]

    distribute_vector=[0]*20

    for re in record:
        distance=distance_sphere(gongcan_lat,gongcan_lon,re[columns['Latitude']]+gongcan_lat,re[columns['Longitude']]+gongcan_lon)

        index=int(distance/15)

        if index>19:
            index=19
        distribute_vector[index]+=1
    
    group_distribute_matrix[key]=np.array(distribute_vector)



#基站道路分布相似度矩阵
distribute_similar_matrix=dict()
for key,vector in group_distribute_matrix.items():

    sim_dic=dict()

    for key_other,vector_other in group_distribute_matrix.items():

        sim = vector_other.dot(vector) / (np.sqrt(vector_other.dot(vector_other)) * (np.sqrt(vector.dot(vector))))

        sim_dic[key_other]=sim

    distribute_similar_matrix[key]=sim_dic



#一些用于统计信息的量
min_dis_count=0
max_dis_count=0

min_sim_count=0
max_sim_count=0

min_distribute_count=0
max_distribute_count=0



#这个重复后来发现并没有什么卵用
for i in range(10):

    other_group_distance_error=dict()
    other_group_sim_error=dict()
    other_group_distribute_error=dict()
    for key in keys:
     
        record = group_rel_coor[key]
 
        error_list=dict()
 
        for key_other,random_forest in forest_dic.items():
 
         
            trainX=[[re[i] for i in index_list] for re in record]
            trainY=[[re[columns['Longitude']],re[columns['Latitude']]]for re in record]
 
            label=random_forest.predict(trainX)
         
            sum = 0
         
            for i in range(len(label)):
 
                sum += distance_sphere(trainY[i][1]+gongcan_2g.纬度[key],trainY[i][0]+gongcan_2g.经度[key],label[i][1]+gongcan_2g.纬度[key],label[i][0]+gongcan_2g.经度[key])
 
            error_list[key_other]=sum/len(label)
 
        distance_vec=distance_matrix[key]
     
        dis_list=[]
        for item in distance_vec:
            dis_list.append(error_list[item])
 
        distribute_matrix=distribute_similar_matrix[key]
        distribute_sort=sorted(distribute_matrix.items(),key=lambda distribute_matrix:distribute_matrix[1],reverse=True)
 
        distribute_list=[]
        for item in distribute_sort:
            distribute_list.append(error_list[item[0]])
 
        sim_matrix=base_station_similar_matrix[key]
        sim_sort=sorted(sim_matrix.items(),key=lambda sim_matrix:sim_matrix[1],reverse=True)
         
        sim_list=[]
        for item in sim_sort:
            sim_list.append(error_list[item[0]])
 
        other_group_distance_error[key]=dis_list
        other_group_distribute_error[key]=distribute_list
        other_group_sim_error[key]=sim_list

    min_dis_error=[]
    max_dis_error=[]
    for key in keys:
        min_dis_error.append(other_group_distance_error[key].index(min(other_group_distance_error[key][1:])))
        max_dis_error.append(other_group_distance_error[key].index(max(other_group_distance_error[key])))

    min_sim_error=[]
    max_sim_error=[]
    for key in keys:
        min_sim_error.append(other_group_sim_error[key].index(min(other_group_sim_error[key][1:])))
        max_sim_error.append(other_group_sim_error[key].index(max(other_group_sim_error[key])))

    min_distribute_error=[]
    max_distribute_error=[]
    for key in keys:
        min_distribute_error.append(other_group_distribute_error[key].index(min(other_group_distribute_error[key][1:])))
        max_distribute_error.append(other_group_distribute_error[key].index(max(other_group_distribute_error[key])))

    for item in min_dis_error:
        if item==1:
            min_dis_count+=1
    for item in min_distribute_error:
        if item==1:
            min_distribute_count+=1
    for item in min_sim_error:
        if item==1:
            min_sim_count+=1

    for item in max_dis_error:
        if item==74:
            max_dis_count+=1
    for item in max_distribute_error:
        if item==74:
            max_distribute_count+=1
    for item in max_sim_error:
        if item==74:
            max_sim_count+=1

