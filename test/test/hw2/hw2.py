from sklearn.cluster import KMeans
import numpy as np
import pandas
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

#轨迹数据栅格化 返回估计数据0/1矩阵
def DataTransform():
    dataGroup = pandas.read_csv("Traj_1000_SH_UTM.csv")

    dataMatrix = dataGroup.as_matrix()

    tid = 1
    #栅格编号是x轴方向上最大的格子数 这样栅格的编号为 (Xi - Xmin)/20 + xmaxNum * (Yi - Ymin)/20
    xmaxNum=(362800-346000)/20

    XYList=[]
    traceList=[]
    #用于存储有记录落在其中的栅格的集合 还可以去重
    cellSet=set()


    for data in dataMatrix:
        if(tid!=data[0]):
            tid +=1
            XYList.append(traceList)
            traceList=[]

        travelX=int((data[2]-346000)/20)
        travelY=int((data[3]-3448600)/20)
        travelIndex=travelX+xmaxNum*travelY

        traceList.append(travelIndex)
        cellSet.add(travelIndex)

    XYList.append(traceList)

    cellList=list(cellSet)
    cellList.sort()

    traceMatrix=[0]*len(cellSet)

    #栅格化后的 0 1 矩阵
    XYMatrix=[]

    #栅格化
    for traceList in XYList:
        for travelIndex in traceList:
            index = cellList.index(travelIndex)
            traceMatrix[index] += 1
        XYMatrix.append(traceMatrix)
        traceMatrix = [0]*len(cellSet)

    return np.array(XYMatrix)

#Kmean最佳分类选择 
def kmeanCluster(DataArray):

    silhouetteList=[]
    X=range(2,50)
    for n_clusters in range(2,50):

        cluster=KMeans(n_clusters=n_clusters,random_state=0)
        cluster_labels=cluster.fit_predict(DataArray)


        score = silhouette_score(DataArray,cluster_labels)
        silhouetteList.append(score)

        print n_clusters
        print score

    plt.plot(X,silhouetteList)
    plt.xlabel("n_clusters")
    plt.ylabel("silhouette")

    plt.show()

    max_index=silhouetteList.index(max(silhouetteList))

    return max_index+2

#DBScan最佳eps选择
def findEPS(DataArray):
    epsLabel = [i/10.0 for i in range(90,120)]
    silhouetteList=[]

    outfile=open("dbscan_2.log","w")

    for eps in epsLabel:
        db=DBSCAN(eps=eps,min_samples=3).fit(DataArray)
        #score=silhouette_score(XYArray,db.labels_)

        labels=db.labels_

        print 1

        n_clusters_=len(set(labels))-(1 if -1 in labels else 0)

        if n_clusters_ > 1:
            
            
            score=silhouette_score(DataArray,[i+1 for i in labels])
            outfile.write(str(eps)+" : "+str(n_clusters_)+" "+str(score)+"\n")
            silhouetteList.append(score)
            print eps
        else:
            silhouetteList.append(-1)

    outfile.close()
        #silhouetteList.append(score)

   
        #print score
    plt.plot(epsLabel,silhouetteList)
    plt.xlabel("eps")
    plt.ylabel("silhouette")

    plt.show()

    max_index=silhouetteList.index(max(silhouetteList))
    return epsLabel[max_index]

#轨迹数据可视化
def writeHTML(clusterType, clusterLabel):

    dataGroup=pandas.read_csv("Traj_1000_SH_GPS_BAIDU_.csv")

    TidList=list(dataGroup['Tid'])
    LonList=list(dataGroup['_Lon'])
    LatList=list(dataGroup['_Lat'])

    colorList=["Blue","Chartreuse","DeepPink","Yellow","Red","Navy","Lime","Aqua"]

    outFile=open(str(clusterType)+".html","w")

    htmlStr="""<!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
        <meta name="viewport" content="initial-scale=1.0, user-scalable=no" />
        <style type="text/css">
            body, html, #allmap {
                width: 100%;
                height: 100%;
                overflow: hidden;
                margin: 0;
                font-family: "微软雅黑";
            }
        </style>
        <script type="text/javascript" src="http://api.map.baidu.com/api?v=2.0&ak=Lcm5jrXRtoBmVZ8bB24G2j4v9pC6LQYQ"></script>
        <title>轨迹可视化结果</title>
    </head>
    <body>
        <div id="allmap"></div>
    </body>
    </html>
    <script type="text/javascript">
	    // 百度地图API功能
	    var map = new BMap.Map("allmap");  // 创建Map实例
	    map.centerAndZoom("上海", 15);      // 初始化地图,用城市名设置地图中心点
	    setTimeout(function () {s
	        map.setZoom(14);
	    }, 2000);  //2秒后放大到14级
	    map.enableScrollWheelZoom(true);

    """
    polyline = "    var polyline1 = new BMap.Polyline([\n"

    index = 0
    tid = 1

    while index < len(TidList):
        if(tid != TidList[index]):
            polyline +="   ], { strokeColor: \"" + colorList[clusterLabel[tid-1]] + "\", strokeWeight: 1, strokeOpacity: 1 });\n    map.addOverlay(polyline" + str(tid) + ")\n"
            
            
            htmlStr += polyline

            tid += 1

            polyline = "    var polyline"+ str(tid) +"= new BMap.Polyline([\n"

        point = "        new BMap.Point("+str(LonList[index])+","+str(LatList[index])+"),\n"
        polyline += point
        index += 1

    polyline +="   ], { strokeColor: \""+colorList[clusterLabel[999]]+"\", strokeWeight: 1, strokeOpacity: 1 });\n    map.addOverlay(polyline" + str(tid) + ")\n"



    htmlStr += polyline

    htmlStr += "</script>"


    outFile.write(htmlStr)

    outFile.close()


    return






#得到特征矩阵进行PCA降维
XYMatrix = DataTransform()
pca=PCA(n_components=1000)
XYMatrix=pca.fit_transform(XYMatrix)


kmeans_class=kmeanCluster(DataArray=XYMatrix)

print "kmeans_class ",kmeans_class


eps=findEPS(DataArray=XYMatrix)

print "eps ",eps

#使用最佳的kmeans_class进行Kmeans聚类
cluster=KMeans(n_clusters=kmeans_class,random_state=0)
cluster_labels=cluster.fit_predict(XYMatrix)
score=silhouette_score(XYMatrix,cluster_labels)
print score
writeHTML(clusterType="Kmean_final",clusterLabel=cluster_labels)

#使用最佳的eps进行DBScan聚类
db=DBSCAN(eps=eps,min_samples=3).fit(XYMatrix)
db_label=np.array([i+1 for i in db.labels_])
score=silhouette_score(XYMatrix,db_label)
print score
writeHTML(clusterType="DBSCAN_final",clusterLabel=db_label)

#Kmeans聚类与GMM聚类进行比较
kmean_classes=len(np.unique(cluster_labels))

#GMM聚类 n_components = kmean_classes
gmm=GaussianMixture(n_components = kmean_classes , max_iter = 20,random_state = 0)
gmm.means_init=np.array([XYMatrix[cluster_labels==i].mean(axis=0) for i in range(kmean_classes)])
gmm.fit(XYMatrix)
gmm_labels=gmm.predict(XYMatrix)
#以Kmeans为基础，计算GMM的准确率
train_accuracy=np.mean(gmm_labels.ravel()==cluster_labels.ravel())*100
print "gmm - kmeans accuracy : ",train_accuracy

#去除DBScan算法认定的噪音
no_noise_matrix=np.array(XYMatrix[db_label!=0])
no_noise_label=np.array(db_label[db_label!=0])


dbscan_class=len(np.unique(no_noise_label))

gmm=GaussianMixture(n_components=dbscan_class,random_state=0)
gmm.means_init=np.array([no_noise_matrix[no_noise_label==i].mean(axis=0) for i in range(dbscan_class)])
gmm.fit(no_noise_matrix)
gmm_labels=gmm.predict(no_noise_matrix)

train_accuracy=np.mean(gmm_labels.ravel()==no_noise_label.ravel())*100
print "gmm - dbscan accuracy : ",train_accuracy

