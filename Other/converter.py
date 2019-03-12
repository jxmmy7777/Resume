import sys
num=16
decimal=3

def numberconverter(num,decimal):
	a=num%decimal
	if num//decimal>0:
		return str(a)+numberconverter(num//decimal,decimal)
	else:
		return str(a)
print numberconverter(16,3)