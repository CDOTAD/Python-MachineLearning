from traHW1 import DataTransform
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np
import pandas
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

XYMatrix=DataTransform()

#DataGroup=pandas.read_csv("Traj_1000_SH_GPS_BAIDU_.csv")

#定义5种用于显示轨迹的颜色
#colorList=['blue','OrangeRed','DeepPink','Indigo','Lime']

XYArray=np.array(XYMatrix)

componentsList=[(1000-i) for i in range(0,1000)]

n_components_choosed=0

for n_components_ in componentsList:
    sum = 0
    pca = PCA(n_components=n_components_)
    transformArray = pca.fit_transform(XYArray)
    for element in pca.explained_variance_ratio_:
        sum+=element
    if sum <0.95:
        n_components_choosed=n_components_+1

        break


print "!!!!!!"

def function():
    X=range(10,26)

    scoreList=[]

    for random_state in range(10,26):
        cluster=KMeans(n_clusters=random_state,random_state=10).fit(XYArray)

        labels=cluster.labels_

        score=silhouette_score(XYArray,labels)
        scoreList.append(score)

        print random_state
        print score
    plt.figure()
    plt.plot(X,scoreList)

    plt.xlabel("n_clusters")
    plt.ylabel("silhouette_score")
    plt.title("silhoutte_score -- n_clusters random_state = 10")
    plt.show()

    return

PCA(n_components=1).fit_transform(XYArray)



#scaler=StandardScaler().fit(XYArray)

#transformMatrix=scaler.transform(XYArray)

epsList = [ i for i in range(1,20)]

outFile = open("eps_1.log","w")

for eps in epsList:
    
    print "!!!!!!!!!!!!!"

    db=DBSCAN(eps=eps,min_samples=10).fit(XYArray)


    labels=db.labels_

    n_clusters_=len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters_>0:
        print eps
        
        outFile.write(str(eps)+" : "+str(n_clusters_)+"\n")

outFile.close()
