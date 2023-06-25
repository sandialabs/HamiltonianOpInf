# Convenience functions for running the ROMs
# Anthony Gruber 3-31-2023

# Standared numpy/scipy imports
import numpy as np
from numpy.linalg import norm, eigvals
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import factorized

# For additional plotting and gifing
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Relative L2 error
def relError(x, xHat):
    num = norm(x - xHat)
    den = norm(x)
    return num / den


def integrate_LFOM(tRange, ics, M, C, K, p, gamma=1./2, beta=1./4):
    N, n_t = M.shape[0], tRange.shape[0]
    h      = tRange[1] - tRange[0]

    M = csc_matrix(M)
    C = csc_matrix(C)
    K = csc_matrix(K)

    q       = np.zeros((N, n_t))
    qDot    = np.zeros((N, n_t))
    qDotDot = np.zeros((N, n_t))
    
    q[:,0], qDot[:,0], qDotDot[:,0] = ics

    LHS        = M + gamma * h * C + beta * h**2 * K
    solve      = factorized(LHS)

    for i,t in enumerate(tRange[:-1]):
        rhst1 = C @ (qDot[:,i] + (1-gamma) * h * qDotDot[:,i])
        rhst2 = K @ (q[:,i] + h * qDot[:,i] + 
                     (0.5-beta) * h**2 * qDotDot[:,i])
        rhs   = p(tRange[i+1]) - rhst1 - rhst2
        
        qDotDot[:,i+1] = solve(rhs)
        qDot[:,i+1]    = qDot[:,i] + h * ((1-gamma) * qDotDot[:,i] 
                                          + gamma * qDotDot[:,i+1])
        q[:,i+1]       = q[:,i] + h * qDot[:,i] \
                                + h**2 * ((0.5-beta) * qDotDot[:,i]
                                        + beta * qDotDot[:,i+1])
    return (q, qDot, qDotDot)


# Function to collect snapshots of Linear FOM using implicit midpoint method
# L is SS mtx, A is mtx rep. of grad H
def integrate_Linear_HFOM(tRange, ic, L, A, safe=False):
    N  = L.shape[0]
    Nt = tRange.shape[0]
    dt = tRange[1] - tRange[0]

    # Store L @ A as sparse mtx
    LA = csc_matrix(L @ A)

    if safe:
        reEigs = eigvals(LA.todense()).real
        if any(reEigs) > 0:
            print('FOM is unstable!')

    # Initialize solution array
    xData      = np.zeros((N, Nt))
    xData[:,0] = ic
    LHS        = identity(N, format='csc') - dt/2 * LA
    solve      = factorized(LHS)

    # Implicit midpoint method
    for i,t in enumerate(tRange[:-1]):
        rhs          = dt * LA @ xData[:,i]
        delXi        = solve(rhs)
        xData[:,i+1] = xData[:,i] + delXi

    # Snapshots of gradH and time derivative Xdot
    gradHdata = A @ xData
    # xDataDot  = FDapprox(xData.T, dt).T
    xDataDot  = LA @ xData

    return xData, xDataDot, gradHdata


# Function which builds operators needed for intrusive ROM
# Only useful when the basis is not blocked.
def build_Linear_ROM_Ops(UU, L, A, ic, n=100, MC=False):
    ic = ic.flatten()

    # Modification for block basis
    # Truncation doesn't work here, so just return stuff...
    if isinstance(UU, list):
        return [[L, A] for i in range(2)]
    else:                       
        U = UU[:,:n]

    # LHat  = -np.linalg.inv(U.T @ L @ U)
    LHat  = U.T @ L @ U
    AHat  = U.T @ A @ U
    LAHat = U.T @ L @ A @ U
    
    x0partHamb4L = np.zeros(n)
    x0PartGal    = np.zeros(n)

    if MC:
        x0partHamb4L += U.T @ A @ ic
        x0PartGal    += U.T @ L @ A @ ic

    HamiltonianOps = [LHat, AHat, x0partHamb4L]
    GalerkinOps    = [LAHat, x0PartGal]

    return HamiltonianOps, GalerkinOps


# Function which integrates a linear ROM using AVF
# Option for Hamiltonian or standard Galerkin
# Option for mean-centering
def integrate_Linear_ROM(tTest, OpList, ic, UU, n, 
                         MC=False, Hamiltonian=False):
    nt = tTest.shape[0]
    dt = tTest[1] - tTest[0]
    ic = ic.reshape(-1,1)

    # Modification for block basis
    # This looks insane, but we can't precompute anything here...
    if isinstance(UU, list):
        Uq = UU[0][:,:n//2]
        Up = UU[1][:,:n//2]
        Zb = np.zeros_like(Uq)
        U  = csc_matrix(np.block([[Uq, Zb], [Zb, Up]]))
        L  = OpList[0]
        A  = OpList[1]
        if Hamiltonian:
            if L.shape[0] == len(ic.flatten()):
                LHat = U.T @ L @ U
                # LHat = -np.linalg.inv(U.T @ L @ U)
            else:
                # If OpList[0] has been replaced with an inferred Lhat
                LHat = L    
            AHat  = U.T @ A @ U
            x0part = np.zeros(n)
            if MC:
                x0part += LHat @ U.T @ A @ ic.flatten()
            LAHat = LHat @ AHat
        else:
            LAHat  = U.T @ L @ A @ U
            x0part = np.zeros(n)
            if MC:
                x0part += U.T @ L @ A @ ic.flatten()
    else:
        U = UU[:,:n]
        if Hamiltonian:
            LHat   = OpList[0][:n,:n]
            AHat   = OpList[1][:n,:n]
            x0b4L  = OpList[2][:n]
            LAHat  = LHat @ AHat
            x0part = LHat @ x0b4L
        else:
            LAHat  = OpList[0][:n,:n]
            x0part = OpList[1][:n]

    # Initialize arrays
    xHat = np.zeros((n, nt))

    # Set initial conditions
    if MC:
        xHat[:,0] = np.zeros(n)
    else:
        xHat[:,0] = U.T @ ic.flatten()

    # Define LHS operators
    LHS     = np.eye(n) - dt/2 * LAHat
    lu, piv = lu_factor(LHS)

    # Integrate ROMs over test interval
    for i,t in enumerate(tTest[:-1]):
        rhs         = dt * (x0part + LAHat @ xHat[:,i])
        delxHati    = lu_solve((lu, piv), rhs)
        xHat[:,i+1] = xHat[:,i] + delxHati

    # Reconstruct FO solutions
    if MC:
        xRec = ic + U @ xHat
    else:
        xRec = U @ xHat

    return xRec


# Function which integrates Generic OpInf or (C or NC) H-OpInf ROM
def integrate_OpInf_ROM(tTest, DhatOp, ic, UU, L=None, approx=False):
    n  = DhatOp.shape[0]
    nt = tTest.shape[0]
    dt = tTest[1] - tTest[0]
    ic = ic.flatten()

    # Modification for block basis
    if isinstance(UU, list):
        Uq   = UU[0][:,:n//2]
        Up   = UU[1][:,:n//2]
        Zb   = np.zeros_like(Uq)
        U    = csc_matrix(np.block([[Uq, Zb], [Zb, Up]]))
    else:                       
        U    = UU[:,:n]

    # Building operators for ROM problem
    xHatOp      = np.zeros((n, nt))
    xHatOp[:,0] = U.T @ ic

    # Define LHS operators
    if L is not None:
        if not approx:
            Lred  = U.T @ L @ U
        else:
            Lred = np.block([[np.zeros((n//2,n//2)), np.eye(n//2)],
                             [-np.eye(n//2), np.zeros((n//2,n//2))]])
        LHSop = np.eye(n) - dt/2 * Lred @ DhatOp
        rhsOp = dt * Lred @ DhatOp
    else:
        LHSop = np.eye(n) - dt/2 * DhatOp
        rhsOp = dt * DhatOp

    # Integrate ROMs over test interval
    lu, piv = lu_factor(LHSop)

    for i,t in enumerate(tTest[:-1]):
        rhs           = rhsOp @ xHatOp[:,i]
        delXhatOpi    = lu_solve((lu, piv), rhs)
        xHatOp[:,i+1] = xHatOp[:,i] + delXhatOpi

    # Reconstruct FO solutions
    XrecOp = U @ xHatOp

    return XrecOp


# Make mp4 movie of an array
def animate_array(arrs, styles, labels, xCoords,
                  eps_x=0.01, eps_y=0.1, yLims=None, legend_loc=0, save=True):
    n_t = arrs[0].shape[-1]
    
    # Define the update function that will be called at each iteration
    def update(lines, data, frame):
        for i,line in enumerate(lines):
            line.set_ydata(data[i][:,frame])
        return lines,

    fig, ax = plt.subplots()

    ## Should fix this to take care of zero....
    xmin = (xCoords.min()*(1-eps_x) if xCoords.min() > 0
            else xCoords.min()*(1+eps_x))
    xmax = (xCoords.max()*(1+eps_x) if xCoords.max() > 0 
            else xCoords.min()*(1-eps_x))
    ymin = (arrs.min()*(1-eps_y) if arrs.min() > 0 else arrs.min()*(1+eps_y))
    ymax = (arrs.max()*(1+eps_y) if arrs.max() > 0 else arrs.max()*(1-eps_y))
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)

    if yLims:
        ax.set_ylim(yLims[0],yLims[1])

    lines = [0 for i in range(len(arrs))]
    for i,line in enumerate(lines):
        lines[i], = ax.plot(xCoords, arrs[i][:,0], 
                            linestyle=styles[i], label=labels[i])
    plt.legend(loc=legend_loc)

    plotFunc = partial(update, lines, arrs)
    anim = FuncAnimation(fig, plotFunc, frames=n_t, interval=20)
    anim.save('myarray.mp4') if save else None
    plt.show()