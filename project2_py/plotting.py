#%% Imports
import matplotlib
import numpy as np
import scipy.stats
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from helpers import Simple1, Simple2
# from project2 import optimize

#%% Modify optimize here to create plots
def optimize(f, g, c, x0, n, count, prob, optfun):
    """
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        c (function): Function evaluating constraints
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `f` and `c` cost 1 and `g` costs 2
        count (function): takes no arguments are reutrns current count
        prob (str): Name of the problem. So you can use a different strategy 
                 for each problem. `prob` can be `simple1`,`simple2`,`simple3`,
                 `secret1` or `secret2`
        optfun (int): either 1 or 2 (for either penalty fn 1 or penalty fn 2)
    Returns:
        x_best (np.array): best selection of variables found
    """

    # TODO: Add Hooke-Jeeves in for the simple problems; try cross-entropy for larger secret problems
    # FOr cross-entropy, denerate random samples with a Gaussian but discard those that violate constraints
    
    # Implement cross-entropy
    nsamp = 100   # number of samples for each distribution
    cov = 4*np.identity(np.size(x0)) # Initial covariance matrix
    nelite = 20 # number of elite samples to pick for new distribution
    c_scale = 20000
    epsilon = 0.01

    # Simple Problem-specific stuff:
    if prob == 'simple1':
        c_scale = 10000
        nsamp = 200
        nelite = 80
        epsilon = 0
    elif prob == 'simple2':
        c_scale = 30000
        nsamp = 200
        nelite = 80
    
    mu = x0
    mus = x0
    while count() < n:
        fs = np.zeros(nsamp)  # Function evaluated at samples
        # Sample from distribution
        samps = np.random.multivariate_normal(mu, cov, nsamp)
        for j in range(nsamp):
            consts = c(samps[j])
            if optfun == 1:
                # fs[j] = f(samps[j]) + c_scale*np.sum(consts > 0)
                fs[j] = f(samps[j]) + c_scale*np.linalg.norm(np.max(consts,0))**2
            elif optfun == 2:
                fs[j] = f(samps[j]) + c_scale*np.sum(np.maximum(consts,0)**2)
        # Retrieve indices with lowest function values
        ind = np.argpartition(fs, nelite)[:nelite]
        # recompute distribution parameters
        cov = np.cov(samps[ind].T)
        mu = np.mean(samps[ind],0)
        mus = np.vstack((mus, mu))
        # print(mu)
    x_best = mu
    print(mus)
    return mus

#%% Initialize vectors

# Initialize instances of simple1, simple2
p1 = Simple1()
p2 = Simple2()

x1 = np.linspace(-3,3)
x2 = np.linspace(-3,3)
simp1_z = np.zeros((50,50))
simp1_c1 = np.zeros((50,50))
simp1_c2 = np.zeros((50,50))
simp2_z = np.zeros((50,50))
simp2_c1 = np.zeros((50,50))
simp2_c2 = np.zeros((50,50))
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
x0s = np.array([[0, 1], [-3,-3], [1,1]])
x,y = np.meshgrid(d,d)
fig, ax = plt.subplots()
plt.imshow( ((x+y**2-1 <=0 ) & (-x-y<=0)).astype(int) , 
                extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap="Greys", alpha = 0.3);
plt.plot(-(x2**2-1), x2)
plt.plot(x1, -x1)
for i in range (3):
    p1_solve = Simple1()
    xs = optimize(p1_solve.f, p1_solve.g, p1_solve.c, x0s[i], p1_solve.n, p1_solve.count, p1_solve.prob, 2)
    plt.plot(xs.T[0], xs.T[1], color="maroon")
    plt.scatter(xs.T[0][-1], xs.T[1][-1], color="red")
    # p1_solve = Simple1()
    # xs = optimize(p1_solve.f, p1_solve.g, p1_solve.c, x0s[i], p1_solve.n, p1_solve.count, p1_solve.prob, 1)
    # plt.plot(xs.T[0], xs.T[1], color="maroon")
    # plt.scatter(xs.T[0][-1], xs.T[1][-1], color="red")
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
x0s = np.array([[-1, 1], [3,-3], [1,2]])
x,y = np.meshgrid(d,d)
fig, ax = plt.subplots()
plt.imshow( (((x-1)**3 - y + 1 <=0 ) & (x + y - 2 <=0)).astype(int) , 
                extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap="Greys", alpha = 0.3);
plt.plot(x1, (x1-1)**3 + 1)
plt.plot(x1, -(x1-2))
for i in range (3):
    p2_solve = Simple2()
    xs = optimize(p2_solve.f, p2_solve.g, p2_solve.c, x0s[i], p2_solve.n, p2_solve.count, p2_solve.prob, 2)
    plt.plot(xs.T[0], xs.T[1], color="maroon")
    plt.scatter(xs.T[0][-1], xs.T[1][-1], color="red")    
    # p2_solve = Simple2()
    # xs = optimize(p2_solve.f, p2_solve.g, p2_solve.c, x0s[i], p2_solve.n, p2_solve.count, p2_solve.prob, 1)
    # plt.plot(xs.T[0], xs.T[1], color="maroon")
    # plt.scatter(xs.T[0][-1], xs.T[1][-1], color="red")
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
# %% Simple2 convergence plots
x0s = np.array([[-1, 1], [1,-1], [1,2]])
f_it = np.zeros((6,3))
maxc_it = np.zeros((6,3))
for i in range(3):
    p2_solve = Simple2()
    xs = optimize(p2_solve.f, p2_solve.g, p2_solve.c, x0s[i], p2_solve.n, p2_solve.count, p2_solve.prob, 1)
    p2 = Simple2()
    # f_it = np.zeros(6)
    # maxc_it = np.zeros(6)
    for j in range(np.shape(xs)[0]):
        f_it[j,i] = p2.f(xs[j])
        maxc_it[j,i] = np.max(p2.c(xs[j]))
plt.plot(f_it)
plt.legend(['x0 = [-1, 1]', 'x0 = [1, -1]', 'x0 = [1, 2]'], bbox_to_anchor=(1.35, 0.6))
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.grid()
plt.figure(figsize=(10,10))
plt.plot(maxc_it)
plt.legend(['x0 = [-1, 1]', 'x0 = [1, -1]', 'x0 = [1, 2]'], bbox_to_anchor=(1.35, 0.6))
plt.xlabel('Iteration')
plt.ylabel('Max Constraint Value')
plt.grid()
plt.figure(figsize=(10,10))

# %%
