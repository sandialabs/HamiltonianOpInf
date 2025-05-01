# utils.py
r"""Utilities for computing errors and plotting figures."""

import os
import opinf
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


DATA_DIRECTORY = "./data"


# Norms and errors ============================================================
def Mnorm(Q: np.ndarray, M, dt: float = 1):
    """Compute the space-time L2 x [t0, tf] norm for a trajectory of states.

    Let M be a mass/weighting matrix such that the spatial L2 norm of y(x, t)
    can be approximated by

        ||y(., t)||_L2 ~= |q(t)|_M = sqrt( q(t)^T M q(t) ),

    where q(t) is a spatial discretization of y(x, t). This function
    approximates the space-time L2 x [t0, tf] norm as

        ||y||^2 ~= sum_{j=0}^{Nt-1} q(tj)^T M q(tj) dt = Nt dt trace(Q^T M Q),

    in which Nt is the number of time instances where the state is measured,
    dt is the spacing between time instances, and Q is the matrix whose columns
    are the state measurements.

    Parameters
    ----------
    Q : (Nx, Nt) ndarray
        State trajectory matrix.
        Each column is the discretized state at a fixed time.
    M : (Nx, Nx) ndarray
        Symmetric, positive definite weight matrix.
    dt : float
        Average spacing between time instances represented in ``Q``.
        This term cancels out when computing relative errors.
    """
    return np.sqrt(Q.shape[1] * dt * np.sum(Q * (M @ Q)))


def solution_error(Q_fom, Q_rom, M=None):
    """Calculate the relative error of two state trajectories, or two
    collections of trajectories, using the Frobenius norm or a weighted norm.

    If ``Q_fom`` and ``Q_rom`` represent individual trajectories (2D arrays),
    calculate the relative error ||Q_fom - Q_rom|| / ||Q_fom||.
    If ``Q_fom`` and ``Q_rom`` each contain several trajectories (3D arrays),
    calculate a joint relative error over all trajectories:

        sqrt( sum_i(||Q_fom[i] - Q_rom[i]||^2) / sum_i(||Q_fom[i]||^2) ).

    This is calculated efficiently by stacking the trajectories horizontally:

        ||hstack(Q_fom) - hstack(Q_rom)|| / ||hstack(Q_fom)||.

    Parameters
    ----------
    Q_fom : (Nx, Nt) or (Ns, Nx, Nt) ndarray
        Trajectory or collection of full-order model trajectories.
    Q_rom : (Nx, Nt) or (Ns, Nx, Nt) ndarray
        Trajectory or collection of reduced-order model trajectories.
    M : (Nx, Nx) ndarray or None
        Weight matrix for the norm. If None (default), use the Frobenius norm;
        otherwise, use ||q|| = q^T M q  <-->  ||Q|| = trace(Q^T M Q).

    Returns
    -------
    relative_error : float
        Relative error of the state trajectory or trajectories.
    """
    if Q_fom.ndim == Q_rom.ndim == 3:
        Q_fom = np.hstack(Q_fom)
        Q_rom = np.hstack(Q_rom)

    if Q_fom.shape != Q_rom.shape or np.any(np.isinf(Q_rom) | np.isnan(Q_rom)):
        return np.nan

    if M is None:
        return la.norm(Q_fom - Q_rom) / la.norm(Q_fom)

    return Mnorm(Q_fom - Q_rom, M) / Mnorm(Q_fom, M)


def projection_error(Q, basis: opinf.basis.PODBasis):
    """Calculate the relative projection error of a state trajectory in a
    given basis, using the Frobenius norm or a weighted norm.

    Parameters
    ----------
    Q : (Nx, Nt) or (Ns, Nx, Nt) ndarray
        Trajectory or collection of full-order model trajectories.
    basis : opinf.basis.PODBasis or models.PSDBasis
        Basis object with a ``project()`` method that maps snapshots to the
        linear subspace spanned by the basis vectors.
    """
    if Q.ndim == 3:
        Q = np.hstack(Q)
    if (M := basis.weights) is None:
        return la.norm(Q - basis.project(Q)) / la.norm(Q)
    return Mnorm(Q - basis.project(Q), M) / Mnorm(Q, M)


# Data management =============================================================
def savenpy(filename, arr):
    """Save a NumPy array to the data directory."""
    if not os.path.isdir(DATA_DIRECTORY):
        os.mkdir(DATA_DIRECTORY)
    np.save(os.path.join(DATA_DIRECTORY, filename), arr)


def loadnpy(filename):
    """Load a NumPy array from the data directory."""
    return np.load(os.path.join(DATA_DIRECTORY, filename))


# Matplotlib configuration and helpers ========================================
def matplotlib_config():
    """Set the matplotlib configuration."""
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc("figure", dpi=300)
    plt.rc("font", size=16, family="serif")
    plt.rc("legend", frameon=False)
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
