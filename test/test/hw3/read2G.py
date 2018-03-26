import pandas
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


def getDataSet_self(fileName):

    raw_data=pandas.read_csv(fileName)

    srncid=np.array(raw_data['SRNCID'])
    best_cellID=np.array(raw_data['BestCellID'])

    dataSet=[srncid,best_cellID]

    rncid_list=[]
    cellid_list=[]
    rssi_list=[]
    rtt_list=[]
    ue_rx_tx_list=[]

    for i in range(6):

        rncid_str='RNCID_'+str(i+1)
        cellid_str='CellID_'+str(i+1)

        ecno_str='EcNo_'+str(i+1)
        rscp_str='RSCP_'+str(i+1)

        rtt_str='RTT_'+str(i+1)
        ue_rx_tx_str='UE_Rx_Tx_'+str(i+1)

        dataSet.append(np.array(raw_data[rncid_str]))
        dataSet.append(np.array(raw_data[cellid_str]))

        dataSet.append(np.array(raw_data[rscp_str])-np.array(raw_data[ecno_str]))

        dataSet.append(np.array(raw_data[rtt_str]))
        dataSet.append(np.array(raw_data[ue_rx_tx_str]))

    label=np.array(raw_data['GridID'])
    dataSet.append(label)
    dataSet=np.array(dataSet)

    return list(dataSet.T)

def getDataSet_2G():

    train2G=getDataSet_self('./hw3/2G GSM/new2gtrain.csv')
    test2G=getDataSet_self('./hw3/2G GSM/new2gtest.csv')

    train2G.extend(test2G)

    return np.array(train2G)

def getDataSet_4G():
    train4G=getDataSet_self('./hw3/4G LTE/new4gtrain.csv')
    test4G=getDataSet_self('./hw3/4G LTE/new4gtest.csv')

    train4G.extend(test4G)

    return np.array(train4G)

def classify(arg):

    data=[]

    if arg=='2G':

        data=getDataSet_2G()
    else:
        data=getDataSet_4G()

    random.shuffle(data2G)

    train=data[:int(len(data)*0.8)]
    test=data[int(len(data)*0.8):]

    train=train.T
    test=test.T

    train_x=(train[0:-1]).T
    train_y=(train[-1]).T

    test_x=(test[0:-1]).T
    test_y=(test[-1]).T

    clf=RandomForestClassifier(n_estimators=25)

    clf.fit(train_x,train_y)

    clf_pre=clf.predict(test_x)

    count=0

    for i in range(len(clf_pre)):
        if clf_pre[i]==test_y[i]:
            count+=1

    ac=float(count)/len(clf_pre)

    return ac

def getDataSet_self_regress(fileName):

    raw_data=pandas.read_csv(fileName)

    srncid=np.array(raw_data['SRNCID'])
    best_cellID=np.array(raw_data['BestCellID'])

    dataSet=[srncid,best_cellID]

    rncid_list=[]
    cellid_list=[]
    rssi_list=[]
    rtt_list=[]
    ue_rx_tx_list=[]

    for i in range(6):

        rncid_str='RNCID_'+str(i+1)
        cellid_str='CellID_'+str(i+1)

        ecno_str='EcNo_'+str(i+1)
        rscp_str='RSCP_'+str(i+1)

        rtt_str='RTT_'+str(i+1)
        ue_rx_tx_str='UE_Rx_Tx_'+str(i+1)

        dataSet.append(np.array(raw_data[rncid_str]))
        dataSet.append(np.array(raw_data[cellid_str]))

        dataSet.append(np.array(raw_data[rscp_str])-np.array(raw_data[ecno_str]))

        dataSet.append(np.array(raw_data[rtt_str]))
        dataSet.append(np.array(raw_data[ue_rx_tx_str]))

    longitude=np.array(raw_data['Longitude'])
    latitude=np.array(raw_data['Latitude'])
    
    dataSet.append(longitude)
    dataSet.append(latitude)

    dataSet=np.array(dataSet)

    return list(dataSet.T)

def getDataSet_2G_regress():

    train2G=getDataSet_self_regress('./hw3/2G GSM/new2gtrain.csv')
    test2G=getDataSet_self_regress('./hw3/2G GSM/new2gtest.csv')

    train2G.extend(test2G)

    return np.array(train2G)

def getDataSet_4G_regress():
    train4G=getDataSet_self_regress('./hw3/4G LTE/new4gtrain.csv')
    test4G=getDataSet_self_regress('./hw3/4G LTE/new4gtest.csv')

    train4G.extend(test4G)

    return np.array(train4G)

def regress(arg):
    data=[]

    if arg=='2G':
        data=getDataSet_2G_regress()
    else:
        data=getDataSet_4G_regress()
    random.shuffle(data)

    train=data[:int(len(data)*0.8)]
    test=data[int(len(data)*0.8):]

    train=train.T
    test=test.T

    train_x=(train[0:-2]).T
    train_y=(train[-2:]).T

    test_x=(test[0:-2]).T
    test_y=(test[-2:]).T

    regr_rf=RandomForestRegressor(max_depth=30,random_state=2)
    regr_rf.fit(train_x,train_y)

    return regr_rf.score(test_x,test_y)
