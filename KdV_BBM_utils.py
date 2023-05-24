# Functions for running the KdV and BBM experiments
# Anthony Gruber 3-31-2023

# Import necessary packages
from functools import partial

import numpy as np
from numpy.linalg import norm, solve
from numpy.fft import rfft, irfft

from scipy.optimize import root
from scipy.integrate import solve_ivp
from scipy.linalg import circulant
from scipy.sparse import csc_matrix, identity, diags
from scipy.sparse.linalg import spsolve

# My own file
from OpInf_utils import FDapprox


# Finite differences: 1st order periodic
def FDforward(y, step):
    diffs      = np.zeros_like(y)
    diffs[:-1] = (y[1:] - y[:-1]) / step
    diffs[-1]  = (y[-1] - y[0]) / step
    return diffs


# Build tridiagonal matrix
def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)


# Newton solve given res, jac functions
# Gets super slow if you crank up the iterations or reduce the tolerance.
def do_newton(res, jac, xOld, tol=1e-6, maxIt=10, verbose=False, sparse=True):
    if not sparse:
        solver = solve
    else:
        solver = spsolve
    xNew = xOld
    err  = norm(res(xNew))
    iter = 0
    while err > tol:
        xNew  = xNew - solver(jac(xNew), res(xNew))
        iter += 1
        if iter > maxIt:
            err  = norm(res(xNew))
            if verbose: print('err =',err)
            break
    return xNew


# Define KdV v1 Hamiltonian (depends on parameters)
def KdV_Hamil_v1(X, dx, a=-6, p=0, v=-1):
    arr = a/6 * X**3 + p/2 * X**2 - v/2 * FDforward(X, dx)**2
    return np.sum(arr, axis=0) * dx


# Define KdV v2 Hamiltonian
def KdV_Hamil_v2(X, dx):
    arr = 0.5 * X**2
    return np.sum(arr, axis=0) * dx


# State-dependent part of L for KdV v2
def C(x, A, a=-6): 
    res = a/3 * np.multiply(A.todense(), x.reshape(-1,1))
    return csc_matrix(res - res.T)


# Generates initial condition for KdV soliton
# Not general -- should probably fix this...
def KdV_soliton_IC(x):
    return 1 / np.cosh(x / np.sqrt(2))**2


# Assemble operators needed for KdV FOM problem in both formulations
def build_KdV_mats(N, xEnds):
    x  = np.linspace(xEnds[0], xEnds[1], N)
    dx = x[1] - x[0]

    # Build derivative matrix A (this is v1 L)
    A       = tridiag(-np.ones(N-1), 
                      np.zeros(N), np.ones(N-1))
    A[-1,0] =  1
    A[0,-1] = -1
    A      *= 1 / (2*dx)
    A       = csc_matrix(A)

    # Build Laplace mtx B
    B       = tridiag(np.ones(N-1),
                      -2*np.ones(N), np.ones(N-1))
    B[-1,0] = 1
    B[0,-1] = 1
    B      *= (1/dx)**2
    B       = csc_matrix(B)

    # Build pentadiagonal circulant matrix
    Ec      = np.zeros(N)
    Ec[1]   = 2
    Ec[2]   = -1
    Ec[-2]  = 1
    Ec[-1]  = -2
    E       = circulant(Ec / 2)
    E      *= (1/dx)**3
    E       = csc_matrix(E)

    return A, B, E


# Function to collect snapshots of KdV v1 FOM
# TIme discretization is AVF
def integrate_KdV_v1_FOM(tRange, ic, A, B, a=-6, p=0, v=-1):
    N  = ic.shape[0]
    Nt = tRange.shape[0]
    dt = tRange[1] - tRange[0]
    
    # Build gradH for KdV v1, depends on central diff mtx B
    def gradH_v1(x, B, a=-6, p=0, v=-1):
        return 0.5*a*x**2 + p*x + v*B@x

    # For root-finding alg
    term2mat = p*identity(N) + v*B
    
    def Nfunc(xOld, xNew):
        xMid  = 0.5 * (xOld + xNew)
        term1 = a/6 * (xOld**2 + xOld*xNew + xNew**2)
        rhs   = A @ (term1 + term2mat @ xMid)
        return xNew - (xOld + dt * rhs)

    def Nderiv(xOld, xNew):
        N     = xNew.shape[0]
        term1 = a/6 * A @ (diags(xOld) + 2*diags(xNew))
        term2 = 1/2 * A @ term2mat
        return identity(N) - dt * (term1 + term2)     

    # Generating snapshots
    Xdata      = np.zeros((N, Nt))
    Xdata[:,0] = ic.flatten()
    
    for i,t in enumerate(tRange[:-1]):
        res  = partial(Nfunc, Xdata[:,i])
        jac  = partial(Nderiv, Xdata[:,i])    
        Xdata[:,i+1] = do_newton(res, jac, Xdata[:,i])

    # Snapshots of gradH and time derivative Xdot
    gradHdata = gradH_v1(Xdata, B, a, p, v)
    XdataDot  = FDapprox(Xdata.T, dt).T

    return Xdata, XdataDot, gradHdata


# Function to collect snapshots of KdV v2 FOM
# Time discretization is AVF
# This version is much slower than Hamil v1
def integrate_KdV_v2_FOM(tRange, ic, A, E, a=-6, p=0, v=-1):
    N  = ic.shape[0]
    Nt = tRange.shape[0]
    dt = tRange[1] - tRange[0]
    
    # For root-finding alg
    term2mat = p*A + v*E
    
    def Nfunc(xOld, xNew):
        xMid  = 0.5 * (xOld + xNew)
        term1 = C(xMid, A, a)
        rhs   = (term1 + term2mat) @ xMid
        return xNew - (xOld + dt * rhs) 

    def Nderiv(xOld, xNew):
        xMid   = 0.5 * (xOld + xNew)
        term1 = C(xMid, A, a)
        term2  = 0.5 * term2mat
        return identity(N) - dt * (term1 + term2)     

    # Generating snapshots
    Xdata      = np.zeros((N, Nt))
    Xdata[:,0] = ic.flatten()
    
    for i,t in enumerate(tRange[:-1]):
        res  = partial(Nfunc, Xdata[:,i])
        jac  = partial(Nderiv, Xdata[:,i])
        Xdata[:,i+1] = do_newton(res, jac, Xdata[:,i])

    # Snapshots of gradH and time derivative Xdot
    gradHdata = Xdata
    XdataDot  = FDapprox(Xdata.T, dt).T

    return Xdata, XdataDot, gradHdata
    

# Precomputing the KdV intrusive ROM operators once and for all
def build_KdV_ROM_Ops(UUlist, A, B, E, ic, n=150, a=-6, p=0, v=-1, MC=False):
    ic      = ic.flatten()
    U1, U2  = UUlist[0][:,:n], UUlist[1][:,:n]
    N       = U1.shape[0]
    
    # For Hamiltonian
    LHat    = U1.T @ A @ U1
    cVecV1  = np.zeros(n)
    cVecV2  = np.zeros(n)
    CmatV1  = U1.T @ (p*identity(N)+v*B) @ U1
    CmatV2  = U2.T @ (p*A + v*E) @ U2
    temp1   = np.einsum('ia,ib->iab', U1, U1)
    TtensV1 = a/2 * np.einsum('ia,ibc', U1, temp1)
    temp2   = np.einsum('ia,ib->iab', U2, U2)
    temp2   = temp2.transpose(1,2,0) @ (A @ U2)
    TtensV2 = a/3 * (temp2.transpose(0,2,1)-temp2.transpose(2,1,0))
    # TtensV2 = a/3 * (temp2 - temp2.transpose(1,2,0))


    # For Galerkin
    cVecV1G  = np.zeros(n)
    cVecV2G  = np.zeros(n)
    CmatV1G  = U1.T @ A @ (p*identity(N)+v*B) @ U1
    CmatV2G  = U2.T @ (p*A + v*E) @ U2
    TtensV1G = a/2 * np.einsum('aj,jbc', U1.T @ A, temp1)

    # Extra terms in case of mean-centering
    if MC:
        # For Hamiltonian
        ich = U2.T @ ic
        cVecV1 += U1.T @ (a/2 * (ic**2) + (p*identity(N)+v*B) @ ic)
        cVecV2 += (U2.T @ C(ic, A, a) @ U2 + CmatV2) @ ich
        CmatV1 += a * U1.T @ (ic.reshape(-1,1) * U1)
        CmatV2 += U2.T @ C(ic, A, a) @ U2 + TtensV2.transpose(0,2,1) @ ich
        # CmatV2 += U2.T @ C(ic, A, a) @ U2 + TtensV2 @ ich

        # For Galerkin
        cVecV1G += U1.T @ A @ (a/2 * ic**2 + (p*identity(N)+v*B) @ ic)
        CmatV1G += U1.T @ A @ (a * ic.reshape(-1,1) * U1)

        cVecV2G += U2.T @ (C(ic, A, a) + p*A + v*E) @ ic
        temp2   = np.einsum('ia,ib->abi', U2, U2)
        # TV2p1   = temp2 @ A.todense()
        TV2p1   = np.einsum('abi,ij', temp2, A.todense())
        TV2p2   = np.einsum('aj,jb->abj', U2.T@A, U2)
        TpartV2 = a/3 * (TV2p1 + TV2p2)
        CmatV2G += TpartV2 @ ic + U2.T @ C(ic, A, a) @ U2


    return ( [cVecV1, CmatV1, TtensV1, LHat], 
             [cVecV1G, CmatV1G, TtensV1G],
             [cVecV2, CmatV2, TtensV2],
             [cVecV2G, CmatV2G, TtensV2] )


# Function to integrate the ROMs for KdV v1
# OpList assumes order of [cVec, Cmat, Ttens, L]
# This function is overloaded for BBM case also
def integrate_KdV_v1_ROM(tTest, OpList, ic, UU, n, MC=False, 
                          Hamiltonian=True, Newton=True):
    nt = tTest.shape[0]
    dt = tTest[1] - tTest[0]
    ic = ic.reshape(-1,1)

    # Building operators for ROM problem
    U     = UU[:,:n]
    cVec  = OpList[0][:n]
    Cmat  = OpList[1][:n,:n]
    Ttens = OpList[2][:n,:n,:n]

    if Hamiltonian:
        LHat = OpList[-1][:n,:n]
    else:
        LHat = np.eye(n)

    # Functions for root finding
    def Nfunc(xHatOld, xHatNew):
        xHatMid   = 0.5 * (xHatOld + xHatNew)
        tensTerm  = ( 2*(Ttens @ xHatOld) @ xHatMid
                  + (Ttens @ xHatNew) @ xHatNew ) / 3
        rhs       = cVec + Cmat @ xHatMid + tensTerm
        return xHatNew - (xHatOld + dt * LHat @ rhs)

    Id = np.eye(n)
    def Nderiv(xHatOld, xHatNew):
        tensTerm  = Ttens @ xHatOld + 2*Ttens @ xHatNew
        return Id - dt * LHat @ (Cmat/2 + tensTerm/3) 

    # Initialize array and set initial conditions
    xHat = np.zeros((n, nt))
    
    if MC:
        xHat[:,0] = np.zeros(n)
    else:
        xHat[:,0] = U.T @ ic.flatten()

    # Integrate FOM/ROMs over test interval
    for i,time in enumerate(tTest[:-1]):
        res = partial(Nfunc, xHat[:,i])
        if not Newton:
            xHat[:,i+1] = root(res, xHat[:,i], method='krylov').x
        else:
            jac = partial(Nderiv, xHat[:,i])  
            xHat[:,i+1] = do_newton(res, jac, xHat[:,i], 
                                    maxIt=3, sparse=False)

    # Reconstruct FO solutions
    if MC:
        xRec = ic + U @ xHat
    else:
        xRec = U @ xHat

    return xRec


# Function to integrate the HROMs for KdV v2
# OpList assumes order of [cVec, Cmat, Ttens]
# Option for standard AVF or "full AVF" integrated
def integrate_KdV_v2_ROM(tTest, OpList, ic, UU, n, MC=False,
                          Newton=False, AVF=True):
    nt = tTest.shape[0]
    dt = tTest[1] - tTest[0]
    ic = ic.reshape(-1,1)
    Id = np.eye(n)

    # Building operators for ROM problem
    U     = UU[:,:n]
    cVec  = OpList[0][:n]
    Cmat  = OpList[1][:n,:n]
    Ttens = OpList[2][:n,:n,:n]

    # Functions for root finding
    def NfuncAVF(xHatOld, xHatNew):
        xHatMid = 0.5 * (xHatOld + xHatNew)
        Tterm   = (Ttens @ xHatMid) @ xHatMid
        rhs     = cVec + Cmat @ xHatMid + Tterm
        return xHatNew - (xHatOld + dt * rhs)

    def NderivAVF(xHatOld, xHatNew):
        xHatMid = 0.5 * (xHatOld + xHatNew)
        Tterm  = (Ttens + Ttens.transpose(0,2,1)) @ xHatMid 
        return Id - dt * (Cmat/2 + Tterm/4)
    
    def NfuncInteg(xHatOld, xHatNew):
        xHatMid = 0.5 * (xHatOld + xHatNew)
        Tterm1  = (Ttens @ xHatNew) @ xHatNew + (Ttens @ xHatOld) @ xHatOld
        Tterm2  = (Ttens @ xHatNew) @ xHatOld + (Ttens @ xHatOld) @ xHatNew
        Tterm   = Tterm1/3 + Tterm2/6
        rhs     = cVec + Cmat @ xHatMid + Tterm
        return xHatNew - (xHatOld + dt * rhs)

    def NderivInteg(xHatOld, xHatNew):
        Tsym  = Ttens + Ttens.transpose(0,2,1)
        Tterm = 1/3 * Tsym @ xHatNew + 1/6 * Tsym @ xHatOld
        return Id - dt * (Cmat/2 + Tterm)

    # Toggle for time integration
    if AVF:
        Nfunc, Nderiv = NfuncAVF, NderivAVF
    else:
        Nfunc, Nderiv = NfuncInteg, NderivInteg

    # Initialize array and set initial conditions
    xHat = np.zeros((n, nt))

    if MC:
        xHat[:,0] = np.zeros(n)
    else:
        xHat[:,0] = U.T @ ic.flatten()

    # Integrate FOM/ROMs over test interval
    for i,time in enumerate(tTest[:-1]):
        res = partial(Nfunc, xHat[:,i])
        if not Newton:
            xHat[:,i+1] = root(res, xHat[:,i], method='krylov').x
        else:
            jac = partial(Nderiv, xHat[:,i])  
            xHat[:,i+1] = do_newton(res, jac, xHat[:,i],
                                    maxIt=10, sparse=False, tol=1e-12)

    # Reconstruct FO solutions
    if MC:
        xRec = ic + U @ xHat
    else:
        xRec = U @ xHat

    return xRec


# This function is adapted from code found at
# https://github.com/W-J-Trenberth/Spectral-method-for-the-BBM-equation
def BBM_solver(t, u_0, a=1, b=1, y=1):
    '''A function to solve the BBM equation 
    $$\partial_t u + a\partial_x u + bu\partial_x u - y\partial_{xxt}u = 0$$.
    Using Fourier analysis this equation can be written in the form 
    $$\partial_t u =F(u)$$. This function uses the fast Fourier transform to 
    compute $F(u)$ quickly solve_ivp to solve the resulting system of ODEs.
    
    Parameters
    -------------------------------------
    t: A time to evaluate the solution at.
    u_0: an array 
    eps: a number representing the strength of the dispersion.
    
    Returns
    ------------------------------------
    out: the solution to BBM evaluated at time t starting from inital data u_0.
    '''
    
    #The RHS of the ODE.
    def RHS(t, u):
        N = len(u)//2 + 1
        n = np.arange(0, N)
        termFT = -2*np.pi*1j*n / (1+y*4*np.pi**2*n**2)
        rhsFT = termFT * rfft(a*u + 0.5*b*u**2)
        return irfft(rhsFT)

    #Use solve_ivp to solve the ODE.
    # xDot = RHS(t, u_0)
    sol = solve_ivp(RHS, [0,t], u_0, t_eval=[t], method='DOP853').y
    return np.reshape(sol, -1)


# Computes the BBM Hamiltonian
def BBM_Hamil(X, dx, a=1, b=1):
    out = 0.5*(a*X**2 + (b/3)*X**3)
    return np.sum(out, axis=0) * dx

def BBM_momentum(X, B, dx, y=1):
    out = X - y * B @ X
    return np.sum(out, axis=0) * dx

def BBM_KinE(X, dx, y=1):
    out = X**2 + y * FDforward(X, dx)**2   #(A @ X)**2
    return 0.5 * np.sum(out, axis=0) * dx


# Precomputing the BBM ROM operators once and for all
def build_BBM_ROM_ops(UU, ic, n=150, MC=False, a=1, b=1):
    U        = UU[:,:n]
    cVec     = np.zeros(n)
    Cmat     = a * np.identity(n)
    temp     = np.einsum('ik,il->ikl', U, U)
    Ttens    = b/2 * np.einsum('ij,ikl', U, temp)

    if MC:
        cVec += U.T @ (a * ic + b/2 * ic**2)
        Cmat += b * U.T @ (ic.reshape(-1,1) * U)

    return [cVec, Cmat, Ttens, np.identity(n)]


# # Function to integrate the OpInf HROM for the BBM equation
# def integrate_BBM_OpInf_HROM(tTest, LHat, TT, ic, UU, n, Newton=True):
#     nt = tTest.shape[0]
#     dt = tTest[1] - tTest[0]

#     # Building operators for ROM problem
#     U        = UU[:,:n]
#     Ttens    = TT[:n,:n,:n]

#     def Nfunc(xHatOld, xHatNew):
#         xHatMid   = 0.5 * (xHatOld + xHatNew)
#         nlin2part = ( 2*(Ttens @ xHatOld) @ xHatMid
#                       + (Ttens @ xHatNew) @ xHatNew ) / 3
#         rhs       = xHatMid + nlin2part
#         return xHatNew - (xHatOld + dt * LHat @ rhs)

#     Id = np.eye(n)
#     def Nderiv(xHatOld, xHatNew):
#         term = Ttens @ xHatOld + 2*Ttens @ xHatNew
#         return Id - dt * (LHat / 2 + LHat @ term / 3)

#     # Initialize array and set initial conditions
#     xHat = np.zeros((n, nt))
#     xHat[:,0] = U.T @ ic.flatten()

#     # Integrate FOM/ROMs over test interval
#     for i,time in enumerate(tTest[:-1]):
#         res = partial(Nfunc, xHat[:,i])
#         if not Newton:
#             xHat[:,i+1] = root(res, xHat[:,i], method='krylov').x
#         else:
#             jac = partial(Nderiv, xHat[:,i])  
#             xHat[:,i+1] = do_newton(res, jac, xHat[:,i],
#                                     maxIt=3, sparse=False)

#     # Reconstruct FO solutions
#     xRec = U @ xHat

#     return xRec