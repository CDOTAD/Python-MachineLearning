import numpy as np
from collections import Counter


dataSet=[]
minSup=0



class ItemSet:
    def __init__(self):
        self.Item=set()
        self.count=0
        return


def search_frequent_1(dataSet):

    itemList=[]
    for i in range(len(dataSet)):
        itemList.extend(dataSet[i])

    count=Counter(itemList)

    large_itemsets=[]
    for key in count.keys():
        if count[key]>=minSup:
            item=ItemSet()
            item.Item.add(key)
            item.count=count[key]
            large_itemsets.append(item)


    return large_itemsets

def is_contain(itemList,itemK):
    for item in itemList:
        if itemK==item.Item:
            return True

    return False

def apriori_gen(itemList_k):
    next_len=len(itemList_k[0].Item)+1
    #print 'next_len ',next_len
    itemList_K=[]
    for i in range(len(itemList_k)):
        for j in range(i+1,len(itemList_k)):

            itemK=[]
            itemK.extend(list(itemList_k[i].Item))
            itemK.extend(list(itemList_k[j].Item))
            itemK=set(itemK)
            if len(itemK)==next_len:

                if not is_contain(itemList_K,itemK):

                    if not is_include_infrenquent_subset(itemList_k,list(itemK)):

                        count=countItem(itemK)
                        if count>=minSup:
                            item=ItemSet()
                            item.Item=set(itemK)
                            item.count=count
                    
                            itemList_K.append(item)
    return itemList_K


def is_include_infrenquent_subset(itemList_k,itemK):
    for i in range(len(itemK)):
        subList=[]
        for j in range(len(itemK)):
            if j==i:
                continue
            else:
                subList.append(itemK[j])

        containFlag=False
        for item in itemList_k:
            if item.Item==set(subList) and item.count>=minSup:
                containFlag=True
                break

        if not containFlag:
            return True


    return False

def countItem(item_k):
    count = 0

    for item in dataSet:
        judge=[False for i in item_k if i not in item]
        if not judge:
            count += 1

    return count

def Apriori(data,sup):

    global dataSet
    global minSup

    dataSet=data
    minSup=sup

    initItemList=search_frequent_1(dataSet)

    for item in initItemList:
        print item.Item,' : ',item.count

    itemList_K=apriori_gen(initItemList)

    
    while itemList_K:
        itemList_k=itemList_K

        for item in itemList_k:
            print item.Item," : ",item.count


        itemList_K=apriori_gen(itemList_k)

    return 