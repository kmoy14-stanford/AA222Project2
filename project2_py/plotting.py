#%% Imports
import matplotlib
import numpy as np
import scipy.stats
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from helpers import Simple1, Simple2

#%% Initialize vectors
x1 = np.linspace(-3,3)
x2 = np.linspace(-3,3)
simp1_z = np.zeros((50,50))
simp1_c1 = np.zeros((50,50))
simp1_c2 = np.zeros((50,50))
simp2_z = np.zeros((50,50))
simp2_c1 = np.zeros((50,50))
simp2_c2 = np.zeros((50,50))
p1 = Simple1()
p2 = Simple2()
for i in range(50):
    for j in range(50):
        x = np.array([x1[i], x2[j]])
        simp1_z[i,j] = p1.f(x)
        simp1_c1[i,j] = p1.c(x)[0]
        simp1_c2[i,j] = p1.c(x)[1]

        simp2_z[i,j] = p2.f(x)
        simp2_c1[i,j] = p2.c(x)[0]
        simp2_c2[i,j] = p2.c(x)[1]

# %% Plot Simple1
d = np.linspace(-3,3,100)
x,y = np.meshgrid(d,d)
fig, ax = plt.subplots()
plt.imshow( ((x+y**2-1 <=0 ) & (-x-y<=0)).astype(int) , 
                extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap="Greys", alpha = 0.3);
plt.plot(-(x2**2-1), x2)
plt.plot(x1, -x1)
plt.legend(['Constraint 1', 'Constraint 2'], bbox_to_anchor=(1.5, 0.6))
# im = ax.imshow(simp1, interpolation='bilinear', origin='lower',
#                cmap=cm.gray, extent=(-3, 3, -3, 3))
CS = ax.contour(x1, x2, simp1_z)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_aspect('equal', adjustable='box')
ax.set_title('Simple1')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.figure(figsize=(10,10))

# %% Plot Simple2

d = np.linspace(-3,3,100)
x,y = np.meshgrid(d,d)
fig, ax = plt.subplots()
plt.imshow( (((x-1)**3 - y + 1 <=0 ) & (x + y - 2 <=0)).astype(int) , 
                extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap="Greys", alpha = 0.3);
plt.plot(x1, (x1-1)**3 + 1)
plt.plot(x1, -(x1-2))
plt.legend(['Constraint 1', 'Constraint 2'], bbox_to_anchor=(1.5, 0.6))
# im = ax.imshow(simp1, interpolation='bilinear', origin='lower',
#                cmap=cm.gray, extent=(-3, 3, -3, 3))
CS = ax.contour(x1, x2, simp2_z, levels = [0, 100, 200, 500, 1000, 2000, 5000, 10000])
ax.clabel(CS, inline=True, fontsize=10)
ax.set_aspect('equal', adjustable='box')
ax.set_title('Simple2')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.figure(figsize=(10,10))
# %%
