# tensor_inference.py
r"""Set up and solve tensor inference problems."""

import numpy as np
import scipy.linalg as la


def infer_Tbar_via_normal_eqns(nus, Ys, Zs):
    """Algorithm 2.1."""

    p, Ns = nus.shape
    r, Nt, Ns = Ys.shape  # or Zs.shape

    Btsr = np.einsum("xs,ias,jas,ys->xijy", nus, Ys, Ys, nus)
    Bhat = Btsr.transpose(0, 1, 3, 2).reshape((r * p, r * p), order="C")

    Ctsr = np.einsum("ias,jas,xs->ijx", Zs, Ys, nus)
    Chat = Ctsr.reshape((r, r * p), order="F")

    Tmat = la.solve(Bhat, Chat.T, assume_a="sym").T
    return Tmat.reshape((r, r, p), order="F")


def infer_Tbar_with_lstsq(nus, Ys, Zs):
    """Algorithm 2.2."""
    p, Ns = nus.shape
    r, Nt, Ns = Ys.shape  # or Zs.shape

    K = np.einsum("ias,xs->iaxs", Ys, nus)
    Dt = K.transpose((0, 2, 1, 3)).reshape((r * p, Nt * Ns), order="F")
    R = Zs.reshape((r, Nt * Ns), order="F")

    Obar = np.linalg.lstsq(Dt.T, R.T)[0].T
    return Obar.reshape((r, r, p), order="F")


def infer_Tbar_with_symmetry(nus, Xs, Ys, Zs, symmetric: bool = True):
    """Algorithm 3.1."""
    p, Ns = nus.shape
    r, Nt, Ns = Ys.shape  # or Zs.shape
    rrp = r * r * p

    XsTXs = np.einsum("kis,kjs->ijs", Xs, Xs)
    YsYsT = np.einsum("iks,jks->ijs", Ys, Ys)
    nusnusT = np.einsum("xs,ys->xys", nus, nus)

    Btsr = np.einsum("xys,ijs,kls->xyijkl", nusnusT, XsTXs, YsYsT)
    Btsr += Btsr.transpose(0, 1, 4, 5, 2, 3)  # tensorized Kronecker sum
    Bhat = Btsr.transpose(0, 2, 4, 1, 3, 5).reshape((rrp, rrp), order="C")

    Ctsr = np.einsum("kis,kas,jas,xs->ijx", Xs, Zs, Ys, nus)
    Ctsr += (1 if symmetric else -1) * Ctsr.transpose(1, 0, 2)
    Chat = Ctsr.flatten(order="F")

    vecT = la.solve(Bhat, Chat, assume_a="sym")
    return vecT.reshape((r, r, p), order="F")
