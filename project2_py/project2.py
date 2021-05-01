#
# File: project2.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project2_py.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np


def optimize(f, g, c, x0, n, count, prob):
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
    Returns:
        x_best (np.array): best selection of variables found
    """
    
    # Implement cross-entropy
    nsamp = 100   # number of samples for each distribution
    cov = 4*np.identity(np.size(x0)) # Initial covariance matrix
    nelite = 20 # number of elite samples to pick for new distribution
    c_scale = 3000

    # Simple Problem-specific stuff:
    if prob == 'simple1':
        c_scale = 10000
        nsamp = 200
        nelite = 80
    elif prob == 'simple2':
        c_scale = 30000
        nsamp = 200
        nelite = 80
    elif prob == 'simple3':
        c_scale = 300
        nsamp = 200
        nelite = 80
    
    mu = x0
    if np.size(x0) > 3: # catch secret2
        nsamp = 100   # number of samples for each distribution
        cov = 0.0001*np.identity(np.size(x0)) # Initial covariance matrix
        nelite = 10 # number of elite samples to pick for new distribution
        c_scale = 10000
        while count() < n:
            fs = np.zeros(nsamp)  # Function evaluated at samples
            # Sample from distribution
            samps = np.random.multivariate_normal(mu, cov, nsamp)
            for j in range(nsamp):
                # Employ count penalty
                consts = c(samps[j])
                fs[j] = f(samps[j]) + c_scale * sum(consts>0)
            # Retrieve indices with lowest function values
            ind = np.argpartition(fs, nelite)[:nelite]
            # recompute distribution parameters
            cov = np.cov(samps[ind].T)
            mu = np.mean(samps[ind],0)
    elif np.size(x0) > 1: #all except secret1
        while count() < n:
            fs = np.zeros(nsamp)  # Function evaluated at samples
            # Sample from distribution
            samps = np.random.multivariate_normal(mu, cov, nsamp)
            for j in range(nsamp):
                consts = c(samps[j])
                fs[j] = f(samps[j]) + c_scale*np.sum(np.maximum(consts,0)**2)
            # Retrieve indices with lowest function values
            ind = np.argpartition(fs, nelite)[:nelite]
            # recompute distribution parameters
            cov = np.cov(samps[ind].T)
            mu = np.mean(samps[ind],0)
    else: # for secret1 only
        # Gradient hyperparameters
        h = 0.01 # stepsize
        x = x0
        # Adam hyperparameters
        gamma_v = 0.99
        gamma_s = 0.999
        alpha = 0.01
        k = 0
        v = 0
        s = 0
        eps = 1e-8
        while count() < n:
            xh = x + h
            # Approximate gradient of function + constants
            consts = c(x)
            constsh = c(xh)
            f_penalty = f(x0) + c_scale*np.sum(np.maximum(consts,0)**2)
            f_penalty_h = f(xh) + c_scale*np.sum(np.maximum(constsh,0)**2)
            g_est = (f_penalty_h - f_penalty)/h
            v = gamma_v*v + (1-gamma_v)*g_est
            s = gamma_s*s + (1-gamma_s)*g_est*g_est
            k += 1
            v_hat = v / (1 - gamma_v**k)
            s_hat = s / (1 - gamma_s**k)
            x = x - alpha * v_hat / (eps + np.sqrt(s_hat))
        mu = x


    x_best = mu

    return x_best