import turtle
def squarePlus(a,b):
    
    for i in range(4):
    	example.fd(a)
    	example.left(90)
    example.penup()
    example.setpos(a/2.0,a/2.0)
    example.left(90)
    example.fd(-b/2.0)
    example.pendown()
    example.fd(b)
    example.right(90)
    example.setpos(a/2.0,a/2.0)
    example.fd(-b/2.0)
    example.fd(b)
    turtle.mainloop()
squarePlus(100,50)