
# coding: utf-8

# In[1]:

import pandas


# In[2]:

train4G=pandas.read_csv('4G/final_4g_tr.csv')


# In[3]:

test4G=pandas.read_csv('4G/final_4g_te.csv')


# In[4]:

columns=dict()


# In[5]:

colunms_list=list(train4G.columns)


# In[6]:

for column in colunms_list:
    
    columns[column]=colunms_list.index(column)


# In[7]:

rscp_list=[columns['RSCP_'+str(i)] for i in range(1,7)]


# In[8]:

ecno_list=[columns['EcNo_'+str(i)]for i in range(1,7)]


# In[9]:

group_divide_grid=dict()


# In[10]:

import numpy as np


# In[12]:

for i in range(len(train4G.Grid_ID)):
    if train4G.Grid_ID[i] not in group_divide_grid.keys():
        
        group_divide_grid[train4G.Grid_ID[i]]=[]
    
    record = list(train4G.ix[i])
    
    re = [record[rscp_list[0]] - record[ecno_list[0]]] 
        
    group_divide_grid[train4G.Grid_ID[i]].append(re)
    


# In[13]:

grid_coordinate=dict()

for i in range(len(train4G.Grid_ID)):
    
    grid_coordinate[train4G.Grid_ID[i]]=[train4G.Grid_center_x[i],train4G.Grid_center_y[i]]




# In[73]:

#baseStation的信息 base_station_list:[[SRNCID,BestCellID]]

base_station_list=[]
for i in range(len(train4G.Grid_ID)):
    
    if [train4G.SRNCID[i],train4G.BestCellID[i]] not in base_station_list:
        
        base_station_list.append([train4G.SRNCID[i],train4G.BestCellID[i]])
        


# In[74]:

len(base_station_list)


# In[129]:
# CellID 为 baseStation在base_station_list中的下标
#与论文中的fingerprint对应 是一个按照GridId分组在按照在按照cellID进行分组 这样{CellID:rssi}和fingerprint相同
# test_divde:{GridID:{CellID:rssiList}}


test_divde=dict()
for i in range(len(train4G.Grid_ID)):
    
    if train4G.Grid_ID[i] not in test_divde.keys():
        
        test_divde[train4G.Grid_ID[i]]=dict()
    
    index = base_station_list.index([train4G.SRNCID[i],train4G.BestCellID[i]])
    
    if index not in test_divde[train4G.Grid_ID[i]].keys():
        
        test_divde[train4G.Grid_ID[i]][index]=[]
    
    record = list(train4G.ix[i])
    
    re = record[ecno_list[0]]-record[rscp_list[0]]
   
    
    test_divde[train4G.Grid_ID[i]][index].append(re)


# In[130]:

test_divde[0]


# In[78]:

rssi_list=[]

for key,records in test_divde.items():
    
    rssi_list.extend(records)


# In[91]:

rssi_list.sort()


# In[188]:
#查看rssi的最大值
max(rssi_list)


# In[92]:
#查看rssi的中值

rssi_list[int(len(rssi_list)/2)]


# In[90]:
#查看rssi的均值
np.mean(rssi_list)


# In[93]:

count = 0

ave=np.mean(rssi_list)

for item in rssi_list:
    
    if item>ave:
        
        count+=1


# In[94]:

count


# In[95]:

len(rssi_list)


# In[96]:

count/float(len(rssi_list))


# In[97]:

rssi_list[int(0.8*len(rssi_list))]


# In[80]:

min(rssi_list)


# In[120]:

np.unique(rssi_list)


# In[131]:
#将rssi映射到0-31之间，这样减少不同的fingerprint的数量。
#在映射公式为(rssi-min)/(max-min) * 31
#在数据处理时发现 min(rssi_list)=0 max(rssi_list)>979 而mean(rssi_list)=71.125 ,中数为69.75
#再通过np.unique(rssi_list)发现这里应该存在离群点，如果之间选min=0,max=979会造成大量数据映射到
#相同的值，而导致无法将fingerprint区分开。所以我们选择将min(rssi_list)选为第一个非零值42.75，
#max(rssi_list)选为排序80%处的rssi值。

min_rssi=42.75
max_rssi=rssi_list[int(0.8*len(rssi_list))]

for key,records in test_divde.items():
        
    for cellID,record in records.items():
        
        rssi_scale=[]
        
        for item in record:
            rssi = round( ((item - min_rssi) / (max_rssi - min_rssi) * 31) ) 

            if rssi<0:

                rssi = 0

            if rssi>31:

                rssi = 31

            rssi_scale.append(rssi)

        test_divde[key][cellID]=rssi_scale


# In[132]:
#检查映射后fingerprint中rssi的值分布情况，看是否是个良好的映射
for key,records in test_divde.items():
    
    for cellID,record in records.items():
        print np.unique(record)


# In[114]:
#fingerprint集合 观察状态集合 observation_symbol:[[CellID,rssi]]
observation_symbol=[]
for key,rssi in test_divde.items():
    
    for cellID,rssi in rssi:
        
        if [key,item] not in observation_symbol:
            
            observation_symbol.append([key,item])


# In[61]:

import math


# In[62]:
#距离计算角度转弧度
def angle2radian(angle):
    
    return math.pi*angle/180.0


# In[63]:
#距离计算函数，用于计算误差
def distance_sphere(Lat1,Lon1,Lat2,Lon2):
    
    R = 6400000
    
    Lat1=angle2radian(Lat1)
    Lat2=angle2radian(Lat2)
    
    Lon1=angle2radian(Lon1)
    Lon2=angle2radian(Lon2)
    
    sita = math.acos(math.sin(Lat1)*math.sin(Lat2) + math.cos(Lat1)*math.cos(Lat2)*math.cos(Lon1-Lon2))
    
    if sita>1:
        sita = 1
    elif sita<-1:
        sita=-1
    
    return R* sita


# In[64]:
#曼哈顿距离计算，其实并没有用，因为发现效果不好，舍弃了
def distance_manhatton(Lat1,Lon1,Lat2,Lon2):
    
    return int(abs(Lat1-Lat2)+abs(Lon1-Lon2))


# In[66]:
#按照ctracker 论文中的trainsition矩阵计算方法进行计算
# T[i][j]=1 (i=j)
# T[i][j]=1/d(i,j)(i!=j) 这里d(i,j)改为球面距离来度量。
#然后再进行归一化
trainsition_score=dict()

for key,coordinate in grid_coordinate.items():
    
    trainsition_score[key]=[]
    
    for other_key,other_coordinate in grid_coordinate.items():
        
        if key == other_key:
            
            trainsition_score[key].append(1)
        
        else:
            
            distance = distance_sphere(coordinate[1],coordinate[0],other_coordinate[1],other_coordinate[0])
            
            trainsition_score[key].append(1.0/distance)


# In[294]:
#转换为trainsition矩阵
trainsition_matrix=np.array([trainsition_score[key] for key in trainsition_score.keys()])


# In[295]:
#查看trainsition_matrix
trainsition_matrix


# In[296]:
# 进行归一化
for i in range(len(trainsition_matrix)):
    
    trainsition_item=trainsition_matrix[i]
    
    sum = 0
    
    for item in trainsition_item:
        
        sum += item
    
    for j in range(len(trainsition_item)):
        
        trainsition_item[j] /= sum


# In[298]:
#检查归一化结果
sum = 0
for item in trainsition_matrix[-1]:
    sum += item
print sum

# In[138]:
#参考ctracker实现的emission矩阵
emission_score=dict()
for gridID,records in test_divde.items():
    
    emission_score[gridID] = [0]*len(observation_symbol)
    
    
    
    for cellID,rssiList in records.items():
        
      
        if len(rssiList)==1:
            
            index = observation_symbol.index([cellID,rssiList[0]])
            
            emission_score[gridID][index] = 1
        else:
            
            for i in range(len(rssiList)):

                ep_list=[]

                for j in range(len(rssiList)):

                    if i == j:
                        continue

                    ep = 3 + (32 - abs(rssiList[i] - rssiList[j]))

                    ep_list.append(ep)


                index = observation_symbol.index([cellID,rssiList[i]])

                emission_score[gridID][index] += max(ep_list)


            
        
    
    


# In[145]:
#转化为矩阵

emission_matrix=np.array([emission_score[key]for key in emission_score.keys()])



# In[292]:
# 进行归一化
for i in range(len(emission_matrix)):
    
    emission_item=emission_matrix[i]
    
    sum = 0
    
    for item in emission_item:
        
        sum += item
    
    for j in range(len(emission_item)):
        
        emission_item[j] /= float(sum)



# In[147]:
# 用于存储第i个出现的GridId
grid_dic=dict()
count = 0
for key in test_divde.keys():
    
    grid_dic[count]=key
    
    count+=1


# In[149]:

from hmmlearn import hmm


# In[151]:

model=hmm.MultinomialHMM(n_components=len(test_divde))


# In[152]:

startprob=np.array([1.0/len(trainsition_matrix)]*len(trainsition_matrix))


# In[153]:

model.startprob_=startprob


# In[301]:

model.transmat_=trainsition_matrix


# In[155]:

model.n_features=len(observation_symbol)


# In[302]:

model.emissionprob_=emission_matrix



#测试集生成
#
# In[160]:

test_label=[]


# In[161]:

test_sequence=[]


# In[163]:

for i in range(len(test4G.Grid_ID)):
    
    if [test4G.SRNCID[i],test4G.BestCellID[i]] not in base_station_list:
        continue
    else:
        
        index = base_station_list.index([test4G.SRNCID[i],test4G.BestCellID[i]])
        
        rssi = test4G.EcNo_1[i] - test4G.RSCP_1[i]
        
        test_sequence.append([index,rssi])
        
        test_label.append(test4G.Grid_ID[i])
        


# In[164]:

len(test_label)


# In[165]:

test_rssi_list=[]
for item in test_sequence:
    
    test_rssi_list.append(item[1])


# In[166]:

min(test_rssi_list)


# In[167]:

max(test_rssi_list)


# In[168]:

test_rssi_list.sort()


# In[169]:

test_rssi_list[int(len(test_rssi_list)/2)]


# In[170]:

np.mean(test_rssi_list)


# In[173]:

test_sequence[0]


# In[175]:
#与训练集同样的方法处理测试集中的rssi
min_test_rssi=min(test_rssi_list)
max_test_rssi=test_rssi_list[int(len(test_rssi_list)*0.8)]

for i in range(len(test_sequence)):
    
    rssi = test_sequence[i][1]
    
    rssi_new = round( ((rssi - min_test_rssi) / (max_test_rssi - min_test_rssi) * 31) )
    
    if rssi_new < 0:
        rssi_new = 0
        
    if rssi_new>31:
        
        rssi_new = 31
    
    test_sequence[i][1]=rssi_new
    
    


# In[176]:

test_sequence[0]


# In[177]:

final_test_sequence=[]
final_test_label=[]
for i in range(len(test_sequence)):
    
    if test_sequence[i] not in observation_symbol:
        continue
    else:
        
        index = observation_symbol.index(test_sequence[i])
        
        final_test_sequence.append([index])
        
        final_test_label.append(test_label[i])


# In[178]:

len(final_test_label)


# In[303]:

result_label=model.predict(final_test_sequence)


# In[304]:

final_result_label=[]


# In[305]:

result_label


# In[306]:

result_label[0]


# In[307]:

for item in result_label:
    
    final_result_label.append(grid_dic[item])


# In[308]:

count = 0
for i in range(len(final_result_label)):
    
    if final_result_label[i]==final_test_label[i]:
        
        count+=1


# In[309]:

count
# count = 1164

# In[310]:

len(final_result_label)
# len = 7558
# GridID 预测准确率 ac = 1164/7558 = 15.4%


# 单层hmm模型建立通过（id ,rssi)直接预测经纬度
# 方法与前面的通过(id, rssi)预测gridID相同
# In[204]:
# 经度列表
lon_list=[]
for item in train4G.Longitude:
    
    lon_list.append(round(item,3))

# In[207]:
# 纬度列表

lat_list=[]
for item in train4G.Latitude:
    
    lat_list.append(round(item,3))

# In[214]:
# 将经度和纬度的精度降低后统计不同的状态数
states_list=[]

# 按照经纬度进行分组统计
coor_divide=dict()

for i in range(len(train4G.Grid_ID)):
    
    lon = round(train4G.Longitude[i],4)
    lat = round(train4G.Latitude[i],4)
    
    if [lon,lat] not in states_list:
        
        states_list.append([lon,lat])
        
    coorID = states_list.index([lon,lat])
    
    if coorID not in coor_divide.keys():
        
        coor_divide[coorID] = dict()
        
    cellID = base_station_list.index([train4G.SRNCID[i],train4G.BestCellID[i]])
    
    if cellID not in coor_divide[coorID]:
        
        coor_divide[coorID][cellID]=[]
    
    rssi = train4G.EcNo_1[i]- train4G.RSCP_1[i]
    
    coor_divide[coorID][cellID].append(rssi)
    
        


# In[215]:

len(states_list)


# In[216]:

states_list[0]


# In[229]:

single_transition=[]

for i in range(len(states_list)):
    
    single_transition.append([0]* len(states_list))
    
    for j in range(len(states_list)):
        
        if i==j:
            
            single_transition[i][j]=1
        else:
            
            single_transition[i][j]=1.0/distance_sphere(states_list[i][1],states_list[i][0],states_list[j][1],states_list[j][0])
            
            


# In[255]:

single_transition_matrix=np.array([item for item in single_transition])


# In[256]:

single_transition_matrix


# In[257]:

for i in range(len(single_transition_matrix)):
    
    single_transition_item = single_transition_matrix[i]
    
    sum = 0
    
    for item in single_transition_item:
        
        sum += item

    
    for j in range(len(single_transition_item)):
        
      
        
        single_transition_item[j] /= sum


# In[258]:

single_transition_matrix


# In[259]:

len(coor_divide)


# In[260]:

coor_divide[0]


# In[261]:

single_rssi_list=[]
for coorID,record in coor_divide.items():
    
    for cellID,rssiList in record.items():
        
        single_rssi_list.extend(rssiList)


# In[262]:

min(single_rssi_list)


# In[263]:

max(single_rssi_list)


# In[265]:

min_rssi=42.75
max_rssi=rssi_list[int(0.8*len(rssi_list))]

for coorID,records in coor_divide.items():
        
    for cellID,record in records.items():
        
        rssi_scale=[]
        
        for item in record:
            rssi = round( ((item - min_rssi) / (max_rssi - min_rssi) * 31) ) 

            if rssi<0:

                rssi = 0

            if rssi>31:

                rssi = 31

            rssi_scale.append(rssi)

        coor_divide[coorID][cellID]=rssi_scale


# In[284]:

single_observation_symbol=[]
for coorID,records in coor_divide.items():
    
    for cellID,rssi in records.items():
        
        for item in rssi:
        
            if [cellID,item] not in single_observation_symbol:
            
                single_observation_symbol.append([cellID,item])


# In[285]:

len(single_observation_symbol)


# In[286]:

single_observation_symbol[0]


# In[287]:

single_emission_score=dict()
for gridID,records in coor_divide.items():
    
    single_emission_score[gridID] = [0]*len(single_observation_symbol)
    
    
    
    for cellID,rssiList in records.items():
        
      
        if len(rssiList)==1:
            
            index = single_observation_symbol.index([cellID,rssiList[0]])
            
            single_emission_score[gridID][index] = 1
        else:
            
            for i in range(len(rssiList)):

                ep_list=[]

                for j in range(len(rssiList)):

                    if i == j:
                        continue

                    ep = 3 + (32 - abs(rssiList[i] - rssiList[j]))

                    ep_list.append(ep)


                index = single_observation_symbol.index([cellID,rssiList[i]])

                single_emission_score[gridID][index] += max(ep_list)


# In[288]:

single_emission_matrix=np.array([single_emission_score[key]for key in single_emission_score.keys()])


# In[289]:

single_emission_matrix


# In[311]:

for i in range(len(single_emission_matrix)):
    
    emission_item=single_emission_matrix[i]
    
    sum = 0
    
    for item in emission_item:
        
        sum += item
    
    for j in range(len(emission_item)):
        
        emission_item[j] /= float(sum)


# In[312]:

single_model=hmm.MultinomialHMM(n_components=len(coor_divide))


# In[321]:

single_startprob=np.array([1.0/len(coor_divide)]*len(coor_divide))


# In[322]:

single_model.startprob_=single_startprob


# In[314]:

len(coor_divide)


# In[315]:

len(single_transition_matrix)


# In[316]:

single_model.transmat_=single_transition_matrix


# In[317]:

single_model.emissionprob_=single_emission_matrix


# In[318]:

single_model.n_features=len(single_observation_symbol)


# In[324]:

single_test_label=[]

single_test_sequence=[]

for i in range(len(test4G.Grid_ID)):
    
    lon = round(test4G.Longitude[i],4)
    
    lat = round(test4G.Latitude[i],4)
    
    if [lon,lat] not in states_list:
        
        continue
    
    elif [test4G.SRNCID[i],test4G.BestCellID[i]] not in base_station_list:
        continue
    else:
        
        index = base_station_list.index([test4G.SRNCID[i],test4G.BestCellID[i]])
        
        rssi = test4G.EcNo_1[i] - test4G.RSCP_1[i]
        
        single_test_sequence.append([index,rssi])
        
        coor_index = states_list.index([lon,lat])
        
        single_test_label.append(coor_index)


# In[325]:

len(single_test_sequence)


# In[326]:

len(final_test_sequence)


# In[323]:

single_result_label = single_model.predict(final_test_sequence)


# In[327]:

single_test_sequence[0]


# In[328]:

single_rssi_list=[]

for item in single_test_sequence:
    
    single_rssi_list.append(item[1])


# In[330]:

min(single_rssi_list)


# In[331]:

max(single_rssi_list)


# In[333]:

min_test_rssi=min(single_rssi_list)
max_test_rssi=single_rssi_list[int(len(single_rssi_list)*0.8)]

for i in range(len(single_rssi_list)):
    
    rssi = single_test_sequence[i][1]
    
    rssi_new = round( ((rssi - min_test_rssi) / (max_test_rssi - min_test_rssi) * 31) )
    
    if rssi_new < 0:
        rssi_new = 0
        
    if rssi_new>31:
        
        rssi_new = 31
    
    single_test_sequence[i][1]=rssi_new


# In[338]:

len(single_test_sequence)


# In[339]:

len(single_test_label)


# In[340]:

final_single_test_sequence=[]
final_single_test_label=[]
for i in range(len(single_test_sequence)):
    
    if single_test_sequence[i] not in single_observation_symbol:
        continue
    else:
        
        index = single_observation_symbol.index(single_test_sequence[i])
        
        final_single_test_sequence.append([index])
        
        final_single_test_label.append(single_test_label[i])


# In[341]:

single_test_result=single_model.predict(final_single_test_sequence)


# In[342]:

count = 0
for pre,tru in zip(single_test_result,single_test_label):
    
    if pre == tru :
        
        count += 1
print count


# In[343]:

single_test_result


# In[344]:

states_list[935]


# In[345]:

single_test_label[0]


# In[346]:

states_list[1637]


# In[347]:
# 距离误差统计
distance_error= 0
for pre,tru in zip(single_test_result,single_test_label):
    
    pre_coor=states_list[pre]
    
    tru_coor=states_list[tru]
    
    distance_error += distance_sphere(pre_coor[1],pre_coor[0],tru_coor[1],tru_coor[0])
    
print distance_error / len(single_test_result)

# 平均距离误差743.768m



