# --coding-- utf-8
#轨迹可视化 输出一个html文件用于轨迹可视化
import pandas
import numpy

dataGroup=pandas.read_csv("Traj_1000_SH_GPS_BAIDU_.csv")

TidList=list(dataGroup['Tid'])

LonList=list(dataGroup['_Lon'])
LatList=list(dataGroup['_Lat'])

colorList=['Crimson','OrangeRed','Navy','Indigo','Lime']

logFile = open('logFile.txt','w')

html=open('visualTrace.html','w')

#html头部信息
htmlStr = """
<!DOCTYPE html>
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
	setTimeout(function () {
	    map.setZoom(14);
	}, 2000);  //2秒后放大到14级
	map.enableScrollWheelZoom(true);

"""

polyline = "    var polyline1 = new BMap.Polyline([\n"

index = 0
tid = 1
#添加轨迹信息 按照csv文件中的GPS信息描绘轨迹 该文件中的GPS坐标经过过GPS坐标和百度坐标转换
while index < len(TidList):
    if(tid != TidList[index]):
        polyline +="   ], { strokeColor: \"blue\", strokeWeight: 1, strokeOpacity: 0.5 });\n    map.addOverlay(polyline" + str(tid) + ")\n"
        tid += 1
        htmlStr += polyline

        polyline = "    var polyline"+ str(tid) +"= new BMap.Polyline([\n"

    point = "        new BMap.Point("+str(LonList[index])+","+str(LatList[index])+"),\n"
    polyline += point
    index += 1

polyline +="   ], { strokeColor: \"blue\", strokeWeight: 1, strokeOpacity: 0.5 });\n    map.addOverlay(polyline" + str(tid) + ")\n"


htmlStr += polyline

htmlStr += "</script>"

html.write(htmlStr)

html.close()

resultNum = 0

