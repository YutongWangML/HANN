import cvxpy as cp
import numpy as np

def get_interior_point(A,b,MaxAbsVal=None):
    D = A.shape[1]
    A1 = np.hstack([-A, np.ones((A.shape[0],1))])
    A2 = np.vstack([np.zeros((D,2)), np.array([-1,1]).T]).T
    A = np.vstack([A1,A2])
    sb = np.array(list(b) + [0,1])
    c = np.array([0 for i in range(D)] + [1])
    x = cp.Variable(D+1)
    if MaxAbsVal is None:
        prob = cp.Problem(cp.Minimize(c.T@x),
                         [A @ x <= sb])
    else:
        prob = cp.Problem(cp.Minimize(c.T@x),
                         [A @ x <= sb,
                         x <= np.abs(MaxAbsVal),
                         x >= -np.abs(MaxAbsVal)])
    prob.solve()
    if x.value is None:
        return None
    else:
        return x.value[:-1]

def grow(s):
    return np.vstack([np.hstack([s, np.ones((s.shape[0],1))]), np.hstack([s, -np.ones((s.shape[0],1))])])

def incremental_enumeration(W,b, MaxAbsVal = None, verbose = True):
    """
    Input: (W,b) is an hyperplane arrangement where
        W    
             Numpy array of the 'slopes', i.e., W.shape = (n_features, n_hyperplanes)
             each column W[:,i] represents the normal vector to the hyperplane
             
        b    
             Numpy array of the 'intercepts', i.e., b.shape = (n_hyperplanes,)
             each entry b[i] represents the bias term of the hyperplane
             
        MaxAbsVal (optional)
             Max value of the interior_pts to be returned. See below.
             
        verbose (optional)
             Print the progress.
             
    Output: (sign_vectors, interior_pts)
        sign_vectors
             Numpy array of the sign vectors
             sign_vectors.shape = (n_sign_vectors, n_hyperplanes)
             The j-th row
             `sign_vectors[j,:]`
             represents the signs of the j-th interior point.
        interior_pts
             Numpy array of the interior points
             interior_pts.shape = (n_sign_vectors, n_hyperplanes)
             `interior_pts[j,:]`
             represents the interior point inside the region corresponding to the j-th sign vector.
             Interior points whose ell-infinity norm (maximum of the absolute value of the largest entry) is larger than MaxAbsVal are ignored.
    """
    sign_vectors = np.array([1,-1])
    sign_vectors = np.expand_dims(sign_vectors,axis=1)
    n_h = 2
    sign_vectors

    while n_h <= len(b):
        interior_pts = []
        sign_vectors_next = []
        sign_vectors = grow(sign_vectors)
        for _s in sign_vectors:
            _S = np.expand_dims(_s, axis = 1)
            interior_pt = get_interior_point(_S * W[:,:n_h].T, _s * b[:n_h], MaxAbsVal)
            if interior_pt is not None:
                interior_pts.append(interior_pt)
                sign_vectors_next.append(_s)
        interior_pts = np.vstack(interior_pts)
        n_h += 1
        sign_vectors = np.vstack(sign_vectors_next)
        if verbose:
            print("Iteration:",n_h-1 , "/", len(b))
            print("Number of cells:", sign_vectors.shape[0])
            print("================")
    return (sign_vectors,interior_pts)