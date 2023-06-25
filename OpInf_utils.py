# Convenience functions for running the various OpInf procedures
# Anthony Gruber 5-28-2023

import numpy as np
from numpy.linalg import solve
from scipy.sparse import csc_matrix, identity, issparse
from scipy.sparse.linalg import spsolve


# Finite differences: 4th order in middle, 1st order at ends
def FDapprox(y, step):
    diffs       = np.zeros_like(y)
    diffs[0]    = (y[1] - y[0]) / step
    diffs[1]    = (y[2] - y[1]) / step
    diffs[2:-2] = (-y[4:] + 8*y[3:-1] - 
                   8*y[1:-3] + y[:-4]) / (12 * step)
    diffs[-2]   = (y[-2] - y[-3]) / step
    diffs[-1]   = (y[-1] - y[-2]) / step
    return diffs


# Vectorize column-wise
def vec(A):
    m, n = A.shape[0], A.shape[1]
    return A.reshape(m*n, order='F')


# Build sparse m x n matrix K such that K @ vec(A) = vec(A.T)
def commutation_matrix_sp(m, n):
    row    = np.arange(m*n)
    col    = row.reshape((m, n), order='F').ravel()
    data   = np.ones(m*n, dtype=np.int8)
    return csc_matrix((data, (row, col)), shape=(m*n, m*n))


# Build precomputable parts of OpInf procedures
# Note that the RHS of canonical OpInf is not truncatable
# This is only useful for non-block basis, when reduced quantities of 
#     lower dimension can be precomputed.
def build_OpInf_stuff(UU, xData, xDotData, gradHData, J,
                      n=150, MC=False):
    
    # Modification for block basis
    # Truncation doesn't work here, so just return stuff
    if isinstance(UU, list):
        if not MC:
            return [[UU, xData, xDotData, gradHData, J] for i in range(3)]
        else:
            return [UU, xData, xDotData, gradHData, J]
    else:
        U    = UU[:,:n]

    xDotHat  = U.T @ xDotData
    gradHatH = U.T @ gradHData
    sgHatH   = gradHatH @ gradHatH.T
    NCrhs    = xDotHat @ gradHatH.T - gradHatH @ xDotHat.T
    NCops    = [sgHatH, NCrhs]

    if not MC: 
        xHat     = U.T @ xData
        xHxHt    = xHat @ xHat.T
        Grhs     = xHat @ xDotHat.T
        Jhat     = U.T @ J @ U
        Cops     = [xHat, xDotHat, xHxHt, Jhat]
        Gops     = [xHxHt, Grhs]
        return   [NCops, Cops, Gops]
    else:   # Recall that mean-centering only makes sense for N-C-OpInf
        return NCops
    

# Infer L in xDot = L gradH(x) 
def NC_H_OpInf(OpList, n, eps=0.):

    # Modification for block basis
    # No fancy tricks, have to compute everything
    if isinstance(OpList[0], list):
        Uq       = OpList[0][0][:,:n//2]
        Up       = OpList[0][1][:,:n//2]
        Zb       = np.zeros_like(Uq)
        U        = csc_matrix(np.block([[Uq, Zb], [Zb, Up]]))
        xDotHat  = U.T @ OpList[2]
        gHatH    = U.T @ OpList[3]
        sgHatH   = gHatH @ gHatH.T
        temp     = xDotHat @ gHatH.T
        rhs      = temp - temp.T
    else:
        # Load precomputed data
        sgHatH   = OpList[0][:n,:n]
        rhs      = OpList[1][:n,:n]

    # Solving NC-H-OpInf Problem
    iMat = np.eye(n)
    P   = csc_matrix( np.kron(iMat, sgHatH) + np.kron(sgHatH, iMat) )
    reg = 2 * eps * identity(n*n)
    Lhat = spsolve(P + reg, vec(rhs)).reshape((n,n), order='F')
    # reg =  eps * identity(n*n)
    # Lhat = spsolve(P @ reg, vec(rhs)).reshape((n,n), order='F')

    return 0.5 * (Lhat - Lhat.T)


# Infer A in xDot = JAx 
# Can add snapshots of forcing if desired (TODO)
# BorisZhu implments "H-OpInf" method from Sharma, Kramer, and Wang (2022)
def C_H_OpInf(OpList, n, Sigma=None, eps=0., approx=True, BorisZhu=False):

    # Modification for block basis
    # No fancy tricks, have to compute everything
    if isinstance(OpList[0], list):
        Uq      = OpList[0][0][:,:n//2]
        Up      = OpList[0][1][:,:n//2]
        Zb      = np.zeros_like(Uq)
        U       = csc_matrix(np.block([[Uq, Zb], [Zb, Up]]))
        xHat    = U.T @ OpList[1]
        xDotHat = U.T @ OpList[2]
        xHxHt   = xHat @ xHat.T
        Jhat    = U.T @ OpList[4] @ U
    else:
        # Load precomputed data
        xHat    = OpList[0][:n]
        xDotHat = OpList[1][:n]
        xHxHt   = OpList[2][:n,:n]
        Jhat    = OpList[3][:n,:n]

    # Only applicable when POD basis comes from SVD of X, as usual
    if Sigma is not None:
        xHxHt = np.diag(Sigma[:n]**2)

    # This is always true if basis comes from symplectic lift
    # Otherwise, this approximation seems to work better....
    if approx:
        JhtJh = np.eye(n)
    else:
        if issparse(Jhat):
            JhtJh = (Jhat.T @ Jhat).todense()
        else:
            JhtJh = Jhat.T @ Jhat

    if BorisZhu:   # implement original H-OpInf method
        # This method relies on a symplectic lift basis
        N = n//2
        xH1xH1t = xHat[:N] @ xHat[:N].T
        xH2xH2t = xHat[N:] @ xHat[N:].T
        temp1   = xDotHat[:N] @ xHat[N:].T
        temp2   = -xDotHat[N:] @ xHat[:N].T
        rhs1    = temp1 + temp1.T
        rhs2    = temp2 + temp2.T

        # Solving the two requisite sub-problems
        P21     = csc_matrix( np.kron(np.eye(N), xH2xH2t) 
                              + np.kron(xH2xH2t, np.eye(N)) )
        P22     = csc_matrix( np.kron(np.eye(N), xH1xH1t) 
                              + np.kron(xH1xH1t, np.eye(N)) )
        A11     = spsolve(P21, vec(rhs1)).reshape((N,N), order='F')
        A22     = spsolve(P22, vec(rhs2)).reshape((N,N), order='F')
        
        # Returning the block diagonal OpInf'd matrix
        Zb      = np.zeros((N,N))
        A       = csc_matrix(np.block([[A22, Zb], [Zb, A11]]))
        return A 
    
    else:   # implement my C-H-OpInf method
        # Can use whatever basis you want
        temp    = Jhat.T @ xDotHat @ xHat.T
        rhs     = temp + temp.T
        P       = csc_matrix( np.kron(JhtJh, xHxHt) + np.kron(xHxHt, JhtJh) )
        reg     = 2 * eps * identity(n*n)
        Ahat    = spsolve(P + reg, vec(rhs)).reshape((n,n), order='F')
        # reg     = eps * identity(n*n)
        # Ahat    = spsolve(P @ reg, vec(rhs)).reshape((n,n), order='F')

        return 0.5 * (Ahat + Ahat.T)
    

# Solving generic OpInf Problem with Willcox method.
# Tikhonov parameter scaled by XX^T
def G_OpInf(OpList, n, Sigma=None, eps=1.0e-15):

    # Modification for block basis
    # Have to recompute everything...
    if isinstance(OpList[0], list):
        Uq      = OpList[0][0][:,:n//2]
        Up      = OpList[0][1][:,:n//2]
        Zb      = np.zeros_like(Uq)
        U       = csc_matrix(np.block([[Uq, Zb], [Zb, Up]]))
        xHat    = U.T @ OpList[1]
        xDotHat = U.T @ OpList[2]
        xHxHt   = xHat @ xHat.T
        rhs     = xHat @ xDotHat.T
    else:
        xHxHt = OpList[0][:n,:n]
        rhs   = OpList[1][:n,:n]

    if Sigma is not None:
        xHxHt = np.diag(Sigma[:n]**2)

    LHS = (1+eps) * xHxHt

    return solve(LHS, rhs).T