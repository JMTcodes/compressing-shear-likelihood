import numpy as np
import matplotlib.pyplot as plt
import time
import sys


m= 1
b = 5
sig =20

x = np.linspace(0,100,num=2001)
y = m*x+b
data_Y = []

for i in range(x.shape[0]):
    print(x[i])
    print(y[i])
    print()
    print()
    data_Y.append(np.random.normal(loc=y[i], scale=sig))

for j in x:
    print(j)

f,(ax1) = plt.subplots(1)
line1 = ax1.scatter(x,data_Y, c="b",s=6)
ax1.grid(True)
ax1.set_ylabel("y")
ax1.set_xlabel("x")
plt.savefig("linear_sim.pdf")
# plt.show()

#csv
f = open('linear_sim.csv', "w")
f.write("x,y\n")

for i in range(len(x)):
    f.write("%s,%s\n" % (x[i],data_Y[i]))

f.close()