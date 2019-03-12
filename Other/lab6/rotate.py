import sys 
def rotate(argv):
    filename1=sys.argv[1]
    filename2=sys.argv[2]
    matrix1=read(filename1)
    matrix2=read(filename2)
    x=(len(matrix1))
    count=0
    matrixnew=[[]for i in range(x)]
    for num1 in range(x):
        for num2 in range(x):
            matrixnew[num1].append(0)
    return(rotate_append(matrix1,matrix2,x,count))



     
   
def rotate_append(matrix1,matrix2,x,count):
        matrixnew=[[]for i in range(x)]
        for num1 in range(x):
            for num2 in range(x):
                matrixnew[num1].append(0)
        for num in range(3):
            for i in range(x):
                for j in range(x):
                    matrixnew[j][x-i-1]=matrix1[i][j]
        if matrixnew==matrix2:
            return True
        else:
            count+=1
            if count==4:
                return False
            else:
                return(rotate_append(matrixnew,matrix2,x,count))

def read(filename):
    inputfile = open(filename,"r")
    data1=inputfile.read()
    data_list=[]
    data_list =data1.split()
    a=[]
    matrix=[]
    for i in data_list:
        a=i.split(",")
        c=[int(x) for x in a]
        matrix.append(c)
    return matrix


print(rotate(sys.argv))
