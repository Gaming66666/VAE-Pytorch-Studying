import turtle as t
#画一个圆形

#速度
t.speed(1) #0最快

#颜色
t.color('purple')

t.circle(160)
t.right(90)
t.circle(180)
t.circle(25)
t.left(15)
t.circle(150)

#循环语句
for i in range(36):
    t.right(9)
    t.circle(60)

for j in range(34):
    t.left(9)
    t.circle(150)

for j in range(34):
    t.left(4)
    t.circle(100)
