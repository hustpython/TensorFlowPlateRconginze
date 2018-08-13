import matplotlib.pyplot as plt 
import numpy as np 
fig, ax = plt.subplots()
plt.title("Loss")
x_ = np.linspace(0,1,300)
y1_ = [-1.0*np.log(i) for i in x_]
y2_ = [np.power((1.0 - i),2) for i in x_]
p1 = ax.plot(x_,y1_,color="r",label='Corss_Entroy')
p2 = ax.plot(x_,y2_,label='L2_Error')
plt.legend()
#plt.show()
plt.savefig("1.png",bbox_inches='tight')
# fig, ax = plt.subplots()
# line1, = ax.plot([1, 2, 3])
# line2, = ax.plot([3, 5, 6])
# line3, = ax.plot([5, 7, 8])

#plt.show()