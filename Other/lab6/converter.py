import sys
num=int(sys.argv[1])
decimal=int(sys.argv[2])#sys is string!!!

def Number_converter(num):
	if decimal<=10:
		a=str(num%decimal)
		if (num/float(decimal))>=1:
			num=num//decimal
			a=a+Number_converter(num)

			return a
		return a
	else:
		if num%decimal<10:
			a=str(num%decimal)
		else:
			a=chr(num%decimal+55)
		if (num/float(decimal))>=1:
			num=num//decimal
			a=a+Number_converter(num)
			return a
		return a




    
result=Number_converter(num)
answer=result[::-1]
print answer