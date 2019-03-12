L=[[],[3,2],[3,3,2,2],4,5,4,3,[2]]
def flatten(L):
    a=[]
    for i in L:
        if type(i)==list:
            a=a+flatten(i)
        else:
            a=a+[i]
    print a
    return a
def frequent(list):
    count=1
    result=1
    list.sort()
    for i in range(len(list)-1):
        if list[i]==list[i+1]:
            count+=1
        else:
            result=max(result,count)
            count=1
    ans=[]
    for i in list:
        if list.count(i)==result:
            if i not in ans:
                ans.append(i)

    print ans,result
frequent(flatten(L))