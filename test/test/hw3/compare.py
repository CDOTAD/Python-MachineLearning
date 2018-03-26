from fp_growth import find_frequent_itemsets
from apriori import Apriori
import matplotlib.pyplot as plt
import datetime
import pandas
import random

test2G=pandas.read_csv('./hw3/2G GSM/new2gtest.csv')

list2G=list(test2G.GridID)

apriori_cost=[]
fp_growth_cost=[]

for time in range(1,6):
        
    random.shuffle(list2G)
    transaction=[]
    for i in range(5):
        transaction.append(list(set(list2G[i*time*10:(i+1)*time*10])))

    apriori_start=datetime.datetime.now()
    
    Apriori(transaction,2)

    apriori_end=datetime.datetime.now()

    apriori_cost.append((apriori_end-apriori_start).microseconds)

    fp_growth_start=datetime.datetime.now()

    for item in find_frequent_itemsets(transaction,2):
        print item

    fp_growth_end=datetime.datetime.now()

    fp_growth_cost.append((fp_growth_end-fp_growth_start).microseconds)


plt.plot(range(5),apriori_cost,'g',label='apriori')
plt.plot(range(5),fp_growth_cost,'r',label='fp-growth')

plt.legend()

plt.show()