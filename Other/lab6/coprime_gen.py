import sys
def coprime_test(argv):
	key=int(argv[1])
	gen_num=int(argv[2])
	num=0
	numbers=key
	a=1
	result=[]
	while num<gen_num:
		numbers+=1
		a=1
		i=0
		for i in range(2,key+1):
			if ((key%i)==0 and ((numbers%i)==0)):
				a=0

		if a==1:
			result.append(str(numbers))
			num+=1
	return result
answer=coprime_test(sys.argv)
print ",".join(answer)

