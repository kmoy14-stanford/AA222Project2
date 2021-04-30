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

    # TODO: Add Hooke-Jeeves in for the simple problems; try cross-entropy for larger secret problems
    # FOr cross-entropy, denerate random samples with a Gaussian but discard those that violate constraints
    
    # Implement cross-entropy
    nsamp = 100   # number of samples for each distribution
    cov = 4*np.identity(np.size(x0)) # Initial covariance matrix
    nelite = 20 # number of elite samples to pick for new distribution
    c_scale = 20000

    # Simple Problem-specific stuff:
    if prob == 'simple1':
        c_scale = 10000
        nsamp = 200
        nelite = 80
    elif prob == 'simple2':
        c_scale = 30000
        nsamp = 250
        nelite = 100
    elif prob == 'simple3':
        c_scale = 300
        nsamp = 200
        nelite = 50
    
    mu = x0
    if np.size(x0) > 3: # catch secret2
        nsamp = 100   # number of samples for each distribution
        cov = 3*np.identity(np.size(x0)) # Initial covariance matrix
        nelite = 30 # number of elite samples to pick for new distribution
        # c_scale = 10000
        gamma = 2
        rho = 2
        #TODO: Interior point method here?? Since x0 is always feasible
        while count() < n:
            fs = np.zeros(nsamp)  # Function evaluated at samples
            # Sample from distribution
            samps = np.random.multivariate_normal(mu, cov, nsamp)
            for j in range(nsamp):
                barrier = 0
                consts = c(samps[j])
                for k in range(np.size(x0)):
                    barrier += 1/(consts[k])
                barrier *= -1/rho
                fs[j] = f(samps[j]) + barrier
            # Retrieve indices with lowest function values
            ind = np.argpartition(fs, nelite)[:nelite]
            # recompute distribution parameters
            cov = np.cov(samps[ind].T)
            mu = np.mean(samps[ind],0)
            # Update interior point scaling
            rho *= gamma
    elif np.size(x0) > 1: #all except secret1
        while count() < n:
            fs = np.zeros(nsamp)  # Function evaluated at samples
            # Sample from distribution
            samps = np.random.multivariate_normal(mu, cov, nsamp)
            for j in range(nsamp):
                fs[j] = f(samps[j]) + c_scale*np.linalg.norm(np.max(c(samps[j]),0))**2
            # Retrieve indices with lowest function values
            ind = np.argpartition(fs, nelite)[:nelite]
            # recompute distribution parameters
            cov = np.cov(samps[ind].T)
            mu = np.mean(samps[ind],0)
    else: # for secret1 only
        spd = 1.0 # standard deviation of distribution
        while count() < n:
            fs = np.zeros(nsamp)
            # sample from distribution
            samps = np.random.normal(x0, spd, nsamp)
            for j in range(nsamp):
                # samp = np.array(samps[j]).reshape((1,1))
                # samp = np.array(samps[j])
                samp = np.array([samps[j]])
                fs[j] = f(samp) + c_scale*np.linalg.norm(np.max(c(samp),0))**2
            # Retrieve indices with lowest function values
            ind = np.argpartition(fs, nelite)[:nelite]
            # recompute distribution parameters
            spd = np.std(samps[ind])
            mu = np.mean(samps[ind])


    x_best = mu

    return x_best