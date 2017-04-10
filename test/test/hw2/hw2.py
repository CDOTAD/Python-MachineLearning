﻿from sklearn.cluster import KMeans
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
    tid = 1
        if(tid != TidList[index]):
            polyline +="   ], { strokeColor: \"" + colorList[clusterLabel[tid-1]] + "\", strokeWeight: 1, strokeOpacity: 1 });\n    map.addOverlay(polyline" + str(tid) + ")\n"
            
            
            htmlStr += polyline

            tid += 1

            polyline = "    var polyline"+ str(tid) +"= new BMap.Polyline([\n"

        point = "        new BMap.Point("+str(LonList[index])+","+str(LatList[index])+"),\n"
        polyline += point
        index += 1

    polyline +="   ], { strokeColor: \""+colorList[clusterLabel[999]]+"\", strokeWeight: 1, strokeOpacity: 1 });\n    map.addOverlay(polyline" + str(tid) + ")\n"

    htmlStr += "</script>"