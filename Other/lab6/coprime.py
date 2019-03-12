import sys
def coprime_test(argv):
	a=0
	integer1=int(argv[1])
	integer2=int(argv[2])
	for i in range(2,max(integer1,integer2)+1):
		if integer1%i==0 and integer2%i==0:
			return False
	
	return True





print(coprime_test(sys.argv))