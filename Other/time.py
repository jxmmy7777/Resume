
import time

start = float(time.time())

def sub_list(L):
    a=len(L)/2
    if len(L)>2:
        L1=L[:a]
        L2=L[a:]
        return sort(sub_list(L1),sub_list(L2))
    elif len(L)==2:
        if L[0]>L[1]:
            L.reverse()
            return L
        return L
    else:
        return L
        
        
def sort(L1,L2):
    L3=[]
    count=0
    for x in L1:
        L3.append(x)
    for i in L2:
        for j in range(count,len(L3)):
            if i<=L3[j]:
                L3.insert(j,i)
                count=j
            elif j==len(L3)-1:
                L3.insert(j+1,i)
    return L3
sub_list([1,5,3,2,4])
end = float(time.time())
elapsed = end - start
print "Time taken: ", elapsed, "seconds."
