from visual import*
g=9.8   #g=9.8
size=0.25 #ball radius
height=2.0 #ball center initial height

scene=display(width=2000,height=2000,center=(-5,1,0),background=(0.5,0.5,0))
floor=box(length=30,height=0.01,width=10,color=color.blue,center=(-5,0,0))
ball=sphere(make_trail=True,radius=size,color=color.red)

ball.pos=vector(-15,height,0)
ball.v=vector(8,8,0)
a1=arrow(shaftwidth=0.1)
a1.color=color.yellow
a1.axis=(0.4,0.4,0)
a1.pos=(-15,height,0)
a1.v=vector(8,8,0)
t=0
dt=0.001
while ball.pos.y>=size:
    rate(1000)
    t+=dt
    ball.pos+=ball.v*dt
    ball.v.y+= -g*dt
    a1.axis=a1.v/20
    a1.pos+=a1.v*dt
    a1.v.y+= -g*dt
    
print t,"seconds"

