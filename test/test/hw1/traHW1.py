#用于轨迹数据栅格化 在利用lshash进行划分 调用writeHtml中的方法进行结果可视化

import pandas
import numpy
import os,sys
#from lshash import LSHash
from sklearn.neighbors import NearestNeighbors



#轨道数据栅格化
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

    return XYMatrix

#编写一个html文件调用百度地图api 将轨道信息可视化 轨道信息会有标记点可以选择添加和删除
def writeHTML( resultList, fileName, queryType ):

    dataGroup=pandas.read_csv("Traj_1000_SH_GPS_BAIDU_.csv")

    TidList=list(dataGroup['Tid'])
    ReverseTid=list(dataGroup['Tid'])
    ReverseTid.reverse()

    TidListLen = len(TidList)

    LonList=list(dataGroup['_Lon'])
    LatList=list(dataGroup['_Lat'])

    #定义5种用于显示轨迹的颜色
    colorList=['blue','OrangeRed','DeepPink','Indigo','Lime']

    logFile = open(fileName+'.log','w')

    html=open(fileName+'.html','w')

    #html 头部

    htmlStr = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
        <meta name="viewport" content="initial-scale=1.0, user-scalable=no" />
        <style type="text/css">
            body, html, #allmap {
                width: 100%;
                height: 95%;
                overflow: hidden;
                margin: 0;
                font-family: "微软雅黑";
            }
            #map-button{
                width: 100%;
                height: 5%;
            }
        </style>
        <script type="text/javascript" src="http://api.map.baidu.com/api?v=2.0&ak=Lcm5jrXRtoBmVZ8bB24G2j4v9pC6LQYQ"></script>
        <script src="http://libs.baidu.com/jquery/1.9.0/jquery.js"></script>
        <title>lshQuery可视化结果</title>
    </head>
    <body>
        <div id="allmap"></div>
        <div id="map-button">
            <input type="button" onclick="add_marker()" value="添加标记点"/>
            <input type="button" onclick="remove_marker()" value="删除标记点"/>
        </div>
    </body>
    </html>
    <script type="text/javascript">
	    // 百度地图API功能
	    var map = new BMap.Map("allmap",{enableMapClick: false});  // 创建Map实例
	    map.centerAndZoom("上海", 15);      // 初始化地图,用城市名设置地图中心点
	    setTimeout(function () {
	        map.setZoom(14);
	    }, 2000);  //2秒后放大到14级
	    map.enableScrollWheelZoom(true);

    """

    #在html中添加编号为15 250 480 690 900的轨迹

    marker1Array="    var marker1Array = new Array()\n"
    marker2Array="    var marker2Array = new Array()\n"
    marker3Array="    var marker3Array = new Array()\n"

    htmlStr += marker1Array
    htmlStr += marker2Array
    htmlStr += marker3Array

    polyline = "    var polyline1 = new BMap.Polyline([\n"
    tid = 1

    for result in resultList:
        for resultIndex in result:
            firstIndex = TidList.index(resultIndex)
            lastIndex =TidListLen - ReverseTid.index(resultIndex)
            

            logFile.write( str(resultIndex)+" : "+str(firstIndex)+" , "+str(lastIndex)+"\n")

            #在轨迹上添加标记点以显示轨迹编号 相似信息

            data_info = "    var data_info =\"轨迹编号:"+str(resultIndex)+" 与轨迹"+str(result[0])+"相近 \";\n"

            marker1 = "    marker1Array.push( new BMap.Marker(new BMap.Point(" +str(LonList[firstIndex]) +","+ str(LatList[firstIndex]) + ")));\n"

            marker2 = "    marker2Array.push( new BMap.Marker(new BMap.Point(" +str(LonList[lastIndex-1]) +","+ str(LatList[lastIndex-1]) + ")));\n"
            
            marker3 = "    marker3Array.push( new BMap.Marker(new BMap.Point(" +str(LonList[(firstIndex+lastIndex)/2]) +","+ str(LatList[(firstIndex+lastIndex)/2]) + ")));\n"   

            htmlStr = htmlStr + marker1 + marker2 + marker3 + data_info 

            htmlStr += "    addClickHandler(data_info,marker1Array[" + str(tid-1) + "]);\n"
            htmlStr += "    addClickHandler(data_info,marker2Array[" + str(tid-1) + "]);\n"  
            htmlStr += "    addClickHandler(data_info,marker3Array[" + str(tid-1) + "]);\n" 
            while firstIndex < lastIndex:
                point = "        new BMap.Point(" + str(LonList[firstIndex]) +","+ str(LatList[firstIndex]) + "),\n"
                polyline += point
                firstIndex += 1

            polyline +="   ], { strokeColor: \""+ str(colorList[resultList.index(result)]) +"\", strokeWeight: 4, strokeOpacity: 1 });\n    map.addOverlay(polyline" + str(tid) + ")\n"
            htmlStr += polyline

            tid += 1

            polyline = "    var polyline"+ str(tid) +"= new BMap.Polyline([\n"

            
    #定义显示信息窗口和函数
    
    handler =""" 

    var opts = {
				width : 250,     // 信息窗口宽度
				height: 80,     // 信息窗口高度
				title : " """+ queryType +"""结果\" , // 信息窗口标题
				enableMessage:true//设置允许信息窗发送短息
			   };


    function addClickHandler(content,marker){
		marker.addEventListener("click",function(e){
			openInfo(content,e)}
		);
	}

    function openInfo(content,e){
		var p = e.target;
		var point = new BMap.Point(p.getPosition().lng, p.getPosition().lat);
		var infoWindow = new BMap.InfoWindow(content,opts);  // 创建信息窗口对象 
		map.openInfoWindow(infoWindow,point); //开启信息窗口
	}  

    """
    #添加marker的JS函数
    add_marker ="""
    function add_marker(){
        for(var i = 0; i< """+str(tid-1)+""";i++){
            map.addOverlay(marker1Array[i]);
            map.addOverlay(marker2Array[i]);
            map.addOverlay(marker3Array[i]);
        }
    }
    """    
    #删除marker的JS函数
    remove_marker ="""
    function remove_marker(){
        for(var i = 0; i<"""+str(tid-1)+""";i++){
            map.removeOverlay(marker1Array[i]);
            map.removeOverlay(marker2Array[i]);
            map.removeOverlay(marker3Array[i]);
        }
    }
    """

    htmlStr += handler

    htmlStr += add_marker
    htmlStr += remove_marker

    htmlStr +="</script>"


    html.write(htmlStr)

    html.close()

    return

#轨道数据进行LSH queryName用于指定记录结果的文件名 hashSize 用于指定lshash的hash_size
"""def traceLSHash(queryName, hashSize):
    #queryName ="hamming_query_12_3"
    #需要进行hashQuery的轨迹index
    indexList=[14,249,479,689,899]

    XYMatrix = DateTransform()

    resultList = []
    nearList = []

    lsh = LSHash(hashSize,44107)
    tid = 1

    for traceList in XYMatrix:
        lsh.index(input_point=traceList,extra_data=tid)
        tid += 1

    resultFile = open(queryName + '.txt','w')

    for index in indexList:
        queryList = lsh.query(XYMatrix[index],distance_func="hamming")
        for result in queryList:
            resultStr = str(index + 1) + " : " + str(result[0][1]) + " " + str(result[1]) +"\n"
            nearList.append(result[0][1])
            resultFile.write(resultStr)
        resultList.append(nearList)
        nearList = [] 

    resultFile.close()

    writeHTML(resultList,queryName, "hashQuerry")
    print resultList

#轨道数据进行KNN queryName用于指定记录结果的文件名 nNeighbors用于指定knn中的nNeighbors
def traceKNN(queryName,nNeighbors):
    indexList=[14,249,479,689,899]

    resultFile = open(queryName + '.txt','w')

    XYMatrix = DateTransform()


    queryMatrix=[]
    for index in indexList:
        queryMatrix.append(XYMatrix[index])


    XYArray = numpy.array(XYMatrix)

    nbrs = NearestNeighbors(n_neighbors=nNeighbors,algorithm='ball_tree').fit(XYArray)

    distances, indices = nbrs.kneighbors(queryMatrix)

    resultList = indices.tolist()

    index = 0
    while index<len(resultList):
        subIndex = 0
        while subIndex<len(resultList[index]):
            resultList[index][subIndex] = resultList[index][subIndex] + 1
            resultFile.write(str(resultList[index][0])+" : "+str(resultList[index][subIndex])+"\n")
            subIndex += 1
        index += 1
            
    resultFile.close()
    writeHTML(resultList,queryName, "KNNQuerry")
    print resultList

    




"""