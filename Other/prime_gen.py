import sys
def prime_gen(key,num):
	t=0
	ans=[]
	count=0
	while count<num:
		for i in range(2,key):
			if key%i==0:
				t=1
				break
			else:
				t=0
		if t==0:
			count+=1
			ans.append(key)
		key+=1
	return ans
key=int(sys.argv[1])
num=int(sys.argv[2])
print prime_gen(key,num)

