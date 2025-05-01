# roms.py
r"""Tensor-based nonintrusive inference of parametric reduced-order models.

These models build off of the ``opinf`` package, in particular the
``opinf.models.ParametricContinuousModel`` class. See the documentation at
https://willcox-research-group.github.io/rom-operator-inference-Python3.
"""

import opinf
import numpy as np
import scipy.linalg as la

import tensor_inference


# ODE models ==================================================================
# Base class ------------------------------------------------------------------
class TensorContinuousModel(opinf.models.ParametricContinuousModel):
    r"""Affine-parametric linear system of ordinary differential equations
    with tensor-based nonintrusive operator inference.

    Parameters
    ----------
    parameter_dimension : int
        Dimension of the parameter vector.
    """

    inference_engine = NotImplemented  # Specified by child classes.

    def __init__(self, parameter_dimension: int):
        """Initialize empty model operators."""
        operators = [opinf.operators.AffineLinearOperator(parameter_dimension)]
        super().__init__(operators)

    def _process_data(self, parameters, states, ddts):
        """Permute data indices to match indexing conventions in the paper."""
        return np.array(parameters).T, np.dstack(states), np.dstack(ddts)

    def _set_entries(self, Ttsr):
        """Set the entries of the affine-parametric linear operator."""
        self.operators[0].set_entries(
            [Ttsr[:, :, i] for i in range(Ttsr.shape[2])], fromblock=False
        )

    def _fit_solver(self, parameters, states, lhs, inputs=None):
        return self.fit(parameters, states, lhs)

    def refit(self):
        return self

    def fit(self, parameters, states, ddts):
        r"""Infer the model operators from data with tensor-based inference.

        Parameters
        ----------
        parameters : list of Ns (floats or (p,) ndarrays)
            Parameter values for which training data are available.
        states : list of Ns (r, Nt) ndarrays
            Snapshot training data. Each array ``states[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``states[i][:, j]`` is a single snapshot.
        ddts : list of s (r, k_i) ndarrays
            Snapshot time derivative data. Each array ``ddts[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``ddts[i][:, j]`` corresponds to the snapshot ``states[i][:, j]``.

        Returns
        -------
        self
        """
        if self.inference_engine is NotImplemented:
            raise TypeError(
                f"Class {self.__class__.__name__} has no `inference_engine`"
            )

        # Re-order indices to match the tensor inference algorithm.
        nus, Ys, Zs = self._process_data(parameters, states, ddts)

        # Infer a tensor.
        Ttsr = self.__class__.inference_engine(nus, Ys, Zs)

        # Unpack the tensor.
        self._set_entries(Ttsr)
        return self

    def predict(self, parameter, state0, t, method="BDF", **options):
        """Integrate the system in time with scipy.integrate.solve_ivp().

        Parameters
        ----------
        parameter : (p,) ndarray
            Parameter instance at which to evaluate the model.
        state0 : (r,) ndarray
            Initial condition for the (reduced) state.
        t : (Nt,) ndarray
            Time domain over which to solve the system.
        method : str
            Type of integrator, see scipy.integrate.solve_ivp().
        **options : dict
            Other keyword arguments for scipy.integrate.solve_ivp().

        Returns
        -------
        state : (r, Nt) ndarray
            System state solution over the specified time domain.
        """
        options["method"] = method
        return super().predict(
            parameter, state0, t, input_func=None, **options
        )


# Models without symmetry preservation ----------------------------------------
class NormalTensorModel(TensorContinuousModel):
    inference_engine = tensor_inference.infer_Tbar_via_normal_eqns


class LstsqTensorModel(TensorContinuousModel):
    inference_engine = tensor_inference.infer_Tbar_with_lstsq


# Models with symmetry preservation -------------------------------------------
class SymmetricTensorModel(TensorContinuousModel):

    inference_engine = tensor_inference.infer_Tbar_with_symmetry

    def fit(self, parameters, states, ddts):
        r"""Infer the model operators from data with tensor-based inference,
        constrained so that the resulting operators are symmetric.

        Parameters
        ----------
        parameters : list of Ns (floats or (p,) ndarrays)
            Parameter values for which training data are available.
        states : list of Ns (r, Nt) ndarrays
            Snapshot training data. Each array ``states[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``states[i][:, j]`` is a single snapshot.
        ddts : list of s (r, k_i) ndarrays
            Snapshot time derivative data. Each array ``ddts[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``ddts[i][:, j]`` corresponds to the snapshot ``states[i][:, j]``.

        Returns
        -------
        self
        """
        # Re-order indices to match the tensor inference algorithm.
        nus, Ys, Zs = self._process_data(parameters, states, ddts)

        # Construct X matrices (Jhat).
        X = np.eye(Ys.shape[0]) if not hasattr(self, "Jhat") else self.Jhat
        Xs = np.dstack([X for _ in range(nus.shape[1])])

        # Infer a symmetric tensor.
        Ttsr = self.__class__.inference_engine(nus, Xs, Ys, Zs, symmetric=True)

        # Unpack the tensor.
        self._set_entries(Ttsr)
        return self


class HamiltonianTensorModel(SymmetricTensorModel):

    def __init__(self, parameter_dimension, basis: opinf.basis.BasisTemplate):
        super().__init__(parameter_dimension)
        self.r = basis.reduced_state_dimension // 2
        Z = np.zeros((self.r, self.r))
        Id = np.eye(self.r)
        self.Jhat = np.block([[Z, Id], [-Id, Z]])
        self.basis = basis

    def predict(self, parameter, state0, t, *args, **kwargs):
        """Integrate the system in time with an energy-conserving scheme.

        Parameters
        ----------
        parameter : (p,) ndarray
            Parameter instance at which to evaluate the model.
        state0 : (r,) ndarray
            Initial condition for the (reduced) state.
        t : (Nt,) ndarray
            Time domain over which to solve the system.

        returns
        -------
        state : (r, Nt) ndarray
            System state solution over the specified time domain.
        """
        # **kwargs is only there for compatibility with the package.
        dt = t[1] - t[0]
        Nt = len(t)

        state = np.empty((2 * self.r, Nt))
        state[:, 0] = state0.reshape(-1)

        A = self.operators[0].evaluate(parameter).entries
        rhsM = la.lu_factor(np.eye(2 * self.r) - (dt / 2) * self.Jhat @ A)

        for i in range(1, Nt):
            state[:, i] = (
                la.lu_solve(rhsM, 2 * state[:, i - 1]) - state[:, i - 1]
            )

        return state


class BlockHamiltonianTensorModel(TensorContinuousModel):
    r"""Affine-parametric linear system of ordinary differential equations
    with tensor-based nonintrusive operator inference.

    This class is for systems with the block structure

            [ dq/dt ]   [ 0  I] [T(mu)  0] [q]
    dy/dt = [ dp/dt ] = [-I  0] [ 0     A] [p]

    where T is a third-order tensor, A is a square matrix, and mu is the '
    parameter vector.

    Parameters
    ----------
    parameter_dimension : int
        Dimension of the parameter vector.
    symmetric : bool
        If ``True`` (default), inferred operators are symmetric.
    """

    def __init__(self, parameter_dimension, symmetric: bool = True):
        self.operator0 = opinf.operators.AffineLinearOperator(
            parameter_dimension
        )
        self.operator1 = opinf.operators.LinearOperator()
        self.__sym = bool(symmetric)
        self.operators = [self.operator0]  # Package compatibility.

    @property
    def symmetric(self):
        """If ``True`` (default), inferred operators are symmetric."""
        return self.__sym

    def split(self, states, ddts):
        """Split up position and momentum data."""
        Qs, Ps, dQs, dPs = [], [], [], []
        for Y, dY in zip(states, ddts):
            Q, P = np.split(Y, 2, axis=0)
            dQ, dP = np.split(dY, 2, axis=0)
            Qs.append(Q)
            Ps.append(P)
            dQs.append(dQ)
            dPs.append(dP)
        return Qs, Ps, dQs, dPs

    def _set_entries(self, Ttsr, A):
        """Set the entries of the affine-parametric linear operator."""
        self.operator0.set_entries(
            [Ttsr[:, :, i] for i in range(Ttsr.shape[2])], fromblock=False
        )
        if A.ndim == 3:
            A = A[..., 0]
        self.operator1.set_entries(A)

    def _fit_symmetric(self, parameters, states, ddts):
        """Infer symmetric operators."""
        # Split the data into position and momentum variables.
        Qs, Ps, dQs, dPs = self.split(states, ddts)

        # Learn the symmetric tensor: dp/dt = -(T mu)q
        nus, Ys, Zs = self._process_data(parameters, Qs, dPs)
        X = np.eye(Ys.shape[0])
        Xs = np.dstack([X for _ in range(nus.shape[1])])
        Ttsr = tensor_inference.infer_Tbar_with_symmetry(nus, -Xs, Ys, Zs)

        # Learn the symmetric matrix: dq/dt = Ap
        ones = np.ones((len(parameters), 1))
        nus, Ys, Zs = self._process_data(ones, Ps, dQs)
        Atsr = tensor_inference.infer_Tbar_with_symmetry(nus, Xs, Ys, Zs)

        # Unpack the entries.
        self._set_entries(Ttsr, Atsr)
        return self

    def _fit_nonsymmetric(self, parameters, states, ddts):
        """Infer operators without enforcing symmetry."""
        # Split the data into position and momentum variables.
        Qs, Ps, dQs, dPs = self.split(states, ddts)

        # Learn the tensor: dp/dt = -(T mu)q
        nus, Ys, Zs = self._process_data(parameters, Qs, dPs)
        Ttsr = -tensor_inference.infer_Tbar_via_normal_eqns(nus, Ys, Zs)

        # Learn the matrix: dq/dt = Ap
        ones = np.ones((len(parameters), 1))
        nus, Ys, Zs = self._process_data(ones, Ps, dQs)
        Atsr = tensor_inference.infer_Tbar_via_normal_eqns(nus, Ys, Zs)

        # Unpack the entries.
        self._set_entries(Ttsr, Atsr)
        return self

    def fit(self, parameters, states, ddts, *args, **kwargs):
        """
        Parameters
        ----------
        parameters : list of Ns (floats or (p,) ndarrays)
            Parameter values for which training data are available.
        states : list of Ns (r, Nt) ndarrays
            Snapshot training data. Each array ``states[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``states[i][:, j]`` is a single snapshot.
        ddts : list of s (r, k_i) ndarrays
            Snapshot time derivative data. Each array ``ddts[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``ddts[i][:, j]`` corresponds to the snapshot ``states[i][:, j]``.
        """
        if self.symmetric:
            return self._fit_symmetric(parameters, states, ddts)
        return self._fit_nonsymmetric(parameters, states, ddts)

    def predict(self, parameter, state0, t, *args, **kwargs):
        """Integrate the system in time with an energy-conserving scheme.

        Parameters
        ----------
        parameter : (p,) ndarray
            Parameter instance at which to evaluate the model.
        state0 : (r,) ndarray
            Initial condition for the (reduced) state.
        t : (Nt,) ndarray
            Time domain over which to solve the system.

        returns
        -------
        state : (r, Nt) ndarray
            System state solution over the specified time domain.
        """
        # *args, **kwargs are there for compatibility with the package.
        dt = t[1] - t[0]
        Nt = len(t)

        # Allocate space for the solution, copy initial condition.
        state0 = state0.reshape(-1)
        r = state0.shape[0] // 2
        r2 = r * 2
        state = np.empty((r2, Nt))
        state[:, 0] = state0

        # Factorize for time stepping.
        Tmu = self.operator0.evaluate(parameter).entries
        A = self.operator1.entries
        Z = np.zeros((r, r))
        Id = np.eye(r2)
        rhsM = la.lu_factor(Id - dt / 2 * np.block([[Z, A], [-Tmu, Z]]))

        # Step forward in time.
        for i in range(1, Nt):
            state[:, i] = (
                2 * la.lu_solve(rhsM, state[:, i - 1]) - state[:, i - 1]
            )

        return state

    def Hamiltonian(self, Yh, mu):
        Qh, Ph = np.split(Yh, 2, axis=0)
        H = np.empty(Yh.shape[1])
        Tmu = self.operator0.evaluate(mu).entries
        A = self.operator1.entries
        for i in range(Yh.shape[1]):
            q, p = Qh[:, i], Ph[:, i]
            H[i] = (q.T @ Tmu @ q) + (p.T @ A @ p)
        return 0.5 * H


# Basis (wave equation with block structure) ==================================
class PSDBasis(opinf.basis.BasisTemplate):
    """Proper symplectic decomposition basis."""

    def __init__(self, name=None, **podbasis_args):
        super().__init__(name=name)
        self.pod = opinf.basis.PODBasis(**podbasis_args)
        W = self.pod.weights
        Z = np.zeros_like(W)
        self.weights = np.block([[W, Z], [Z, W]])

    def split(self, states):
        """Split up position and momentum data."""
        return np.split(states, 2, axis=0)

    def join(self, Q, P):
        """Join position and momentum data."""
        return np.concatenate((Q, P), axis=0)

    def fit(self, states):
        self.pod.fit(np.hstack(self.split(states)))
        self.full_state_dimension = 2 * self.pod.full_state_dimension
        self.reduced_state_dimension = 2 * self.pod.reduced_state_dimension
        return self

    def compress(self, states):
        Q, P = self.split(states)
        return self.join(self.pod.compress(Q), self.pod.compress(P))

    def decompress(self, states_compressed, locs=None):
        Q_, P_ = self.split(states_compressed)
        return self.join(self.pod.decompress(Q_), self.pod.decompress(P_))

    def set_dimension(self, **podbasis_args):
        """NOTE: num_vectors=r indicates the basis size for each variable,
        so the total reduced state dimension would be 2r.
        """
        self.pod.set_dimension(**podbasis_args)
        self.reduced_state_dimension = 2 * self.pod.reduced_state_dimension
