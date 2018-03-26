import numpy as np

from collections import Counter

data=[]

FP_root=TreeNode()
FP_root.name=None
FP_root.count=None

large_itemsets=[]

minSup=0

class Item:
    def __init__(self,name,count):

        self.name=name
        self.count=count
        return


    def __gt__(self,other):
        return self.count>other.count

    def __lt__(self,other):
        return self.count<other.count

    def __str__(self):
        return '(%d, %d)'%(self.name,self.count)

class TreeNode:

    def __init__(self):
        self.name=0
        self.count=0
        self.child=[]

        return

class Pattern:

    def __init__(self):
        self.ItemSet=set()
        self.support=0
        return

def search_frequnt_1(data):

    itemList=[]
    for item in data:
        itemList.extend(item)

    count=Counter(itemList)

    large_itemsets=[]

    for key in count.keys():
        item=Item(key,count[key])
        large_itemsets.append(item)

    return large_itemsets

def index_in_sortedlist(item):

    for i in range(len(large_itemsets)):
        if item == large_itemsets[i].name:
            return i

    return -1

def transaction_path_sort(list,start_index,end_index):
    flag = index_in_sortedlist(list[end_index])
    i = start_index - 1
    for j in range(start_index,end_index):
        if index_in_sortedlist(list[j]) > flag:
            pass
        else:
            i += 1
            tmp = list[i]
            list[i] = list[j]
            list[j] = tmp
    tmp = list[end_index]
    list[end_index] = list[i+1]
    list[i+1] = tmp

    return i+1

def transaction_quick_sort(list,start_index,end_index):
    if start_index>=end_index:
        return
    middle=transaction_path_sort(list,start_index,end_index)
    transaction_quick_sort(list,start_index,middle-1)
    transaction_quick_sort(list,middle+1,end_index)

    return

def transaction_sort():

    for item in data:
        transaction_quick_sort(item,0,len(item)-1)

    return

def index_in_father(fatherNode,item):
    
    for i in range(len(fatherNode.child)):
        if fatherNode.child[i].name==item:
            return i

    return -1

def add_tree_node(rootNode,item):

    if not item:
        return

    index=index_in_father(rootNode,item[0])
    if index ==-1:

        fatherNode=rootNode
        for i in item:

            childNode=TreeNode()
            childNode.name=i
            childNode.count=1

            fatherNode.child.append(childNode)
            fatherNode=childNode


    else:

        rootNode.child[index].count += 1
        add_tree_node(rootNode.child[index],item[1:])



    return

def build_FPtree():

    for transaction in data:
    
        add_tree_node(FP_root,transaction)

    return

def freq_pattern_set(tree):

    return

def FP_growth_kernal(tree,item):
    P=TreeNode()
    Q=TreeNode()
    if len(tree.child)==1:
        P.name=None
        P.count=None

        
        fatherNode=P

        

        childNode=tree
        while len(childNode.child)==1:

           
            p_childNode=TreeNode()
            p_childNode.name=childNode.child[0].name
            p_childNode.count=childNode.child[0].name

            fatherNode.child.append(p_childNode)
            fatehrNode=p_childNode

            childNode=childNode.child[0]

        Q=childNode
        Q.name=None
        Q.count=None
    else:
        Q=tree




