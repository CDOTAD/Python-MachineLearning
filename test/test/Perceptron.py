#感知机

LearningRate = 1

class Vector:
    
    def __init__( self , x1 , x2 , b ):
        self.x1 = x1
        self.x2 = x2
        self.b = b

    def DotProduct(self,other):
        return self.x1*other.x1+self.x2*other.x2+self.b*other.b

    def Multiply(self,num):
        return Vector(self.x1*num,self.x2*num,self.b*num)

    def Add(self,other):
        return Vector(self.x1+other.x1,self.x2+other.x2,self.b+other.b)

    def __str__(self):
        return '(%d,%d,%d)'%(self.x1,self.x2,self.b)


class Data:

    def __init__( self , x1 , x2 , y ):
        self.vector=Vector(x1,x2,1)
        self.y = y


W=Vector(0,0,0)

data1 = Data(3,3,1)
data2 = Data(4,3,1)
data3 = Data(1,1,-1)

DataSet=(data1,data2,data3)

flag=True
finish=False


while flag:
    finish=True
    for data in DataSet:
        if (data.vector.DotProduct(W))*data.y<=0:
            finish=False
            addVector = data.vector.Multiply(LearningRate*data.y)
            W=W.Add(addVector)
            break
    if finish:
        flag=False

    


print W;






