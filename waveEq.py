# waveEq.py
"""Models for the wave equation with piecewise constant wave speed."""

__all__ = [
    "WaveFEM",
    "WaveFEM1D",
    "WaveFEM2D",
]

import abc
import numpy as np
import IPython.display
import scipy.sparse as sp
import matplotlib.animation
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla

import opinf
import ngsolve as ng
from ngsolve.webgui import Draw
from ngsolve.meshes import Make1DMesh
from netgen.geom2d import SplineGeometry


from scipy.linalg import inv


# Full-order finite-element-based models ======================================
class WaveFEM(abc.ABC):
    r"""Base class for mixed finite element models of the wave equation with a
    piecewise constant wave speed and homogeneous Dirichlet boundary
    conditions. The semi-discrete system has the block form

        [ M_W   0  ] [ dq/dt ]   [  0  I ] [ A(μ)   0  ] [ q ]
        [  0   M_W ] [ dp/dt ] = [ -I  0 ] [  0    M_W ] [ p ],

    where A(μ) has an affine parametric decomposition

        A(μ) = 1/μ_1^2 A_1 + ... + 1/μ_p^2 A_p \approx S^T M_V(μ)^{-1} S.

    Here, M_W, M_V are mass matrices of the two finite element spaces, and S
    is the coupling matrix.

    The initial condition and the underlying spatial dimension and geometry
    are specified in child classes.

    Parameters
    ----------
    orderW : int
        Polynomial order of the L2 finite element space.
    orderV : int
        Polynomial order of the H-Div finite element space.
    """

    dim = NotImplemented
    dirichlet_BCs = NotImplemented
    parameter_dimension = NotImplemented

    def __init__(self, orderW: int = 1, orderV: int = 1):
        # Initialize the finite element space
        self.mesh = self.create_mesh()
        self.W = ng.L2(self.mesh, order=orderW)
        if self.dim == 1:
            self.V = ng.H1(self.mesh, order=orderV)
        else:
            self.V = ng.HDiv(self.mesh, order=orderV, RT=True)

        # Get the mass matrix associated with space W
        M = ng.BilinearForm(self.W)
        M += self.W.TrialFunction() * self.W.TestFunction() * ng.dx
        self.MW = self._tocoo(M).tocsc()
        self.MWinv = spla.inv(self.MW)

        # Get the S matrix
        self.S = self._tocoo(self.getS()).tocsc()

        # Get the mass matrices associated with space V
        self.MVs = self.getMVs()

        # ICs
        self.q0 = self._asarray(self.initial_condition()[0])
        self.p0 = self._asarray(self.initial_condition()[1])

    @property
    def Nx(self) -> int:
        """Size of the spatial discretization"""
        return self.W.ndof

    @property
    def nodes(self) -> np.ndarray:
        """Nodes in the spatial mesh"""
        return self.getNodes()

    def __str__(self):
        """String representation."""
        interval = f"[0, {self.L:.4f}]"
        domain = " x ".join([interval] * self.dim)
        return "\n".join(
            [
                "Parametric wave equation finite element model",
                f"  Spatial domain ({self.dim:d}D): {domain}",
                f"  Discretization size: {self.Nx}",
            ]
        )

    # Abstract methods --------------------------------------------------------
    @abc.abstractmethod
    def create_mesh(self):
        """Create the spatial finite element mesh."""
        raise NotImplementedError

    @abc.abstractmethod
    def initial_condition(self, asarray: bool = True):
        """Construct the initial condition.

        Parameters
        ----------
        asarray : bool
            If ``True`` (default), return a NumPy array.
            If ``False``, return an ``ngsolve`` object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getS(self):
        """Get the coupling matrix corresponding to the mixed bilinear form."""
        raise NotImplementedError

    @abc.abstractmethod
    def getMVs(self) -> list:
        """Assemble the (parameter-independent) matrices (M_V)_1, ..., (M_V)_p
        used in constructing the parameter-dependent matrix A(μ).
        """
        raise NotImplementedError

    # Utilities ---------------------------------------------------------------
    def getNodes(self) -> np.ndarray:
        """Returns the coordinates of the spatial nodes"""
        pnts_x = []
        for e in self.mesh.edges:
            for v in e.vertices:
                pnts_x.append(self.mesh[v].point[0])
        if self.W.globalorder == 0:
            return np.array(pnts_x)[::2] / 2 + np.array(pnts_x)[1::2] / 2
        return pnts_x

    def _tocoo(self, A) -> sp.coo_array:
        """Convert an ngsolve matrix to a SciPy sparse COO array."""
        coo = A.Assemble().mat.COO()
        return sp.coo_array((coo[2], coo[:2])).tocsc()

    def _asarray(self, func):
        """Convert an ngsolve function to a NumPy array."""
        f = ng.GridFunction(self.W)
        f.Set(func)
        return f.vec.FV().NumPy()

    def sample_parameters(
        self,
        low: float,
        high: float,
        num_samples: int = 10,
        train_ratio: float = 0.8,
        randseed: int = 0,
    ):
        """Sample the parameter space to generate training and testing sets.

        Parameters
        ----------
        low : float
            Lower bound for parameter sampling.
        high : float
            Upper bound for parameter sampling.
        num_samples : int
            Total number of parameter vectors.
        train_ratio : float
            Proportion of the parameter vectors to be used for training.
        randseed : int
            Random seed for the sampling.

        Returns
        -------
        training_parameters : (N_train, parameter_dimension) ndarray
            Parameter vectors to train on.
        testing_parameters : (N_test, parameter_dimension) ndarray
            Parameter vectors to test on.
        """
        np.random.seed(randseed)

        # Sample from the parameter space.
        samples = np.random.uniform(
            low=low,
            high=high,
            size=(num_samples, self.parameter_dimension),
        )

        # Split into training and testing sets.
        split = int(train_ratio * num_samples)
        return samples[:split], samples[split:]

    def save_VTK(self, q, filename):
        """Save the solution to a VTK file."""
        gfu = ng.GridFunction(self.W)
        gfu.vec.FV().NumPy()[:] = q
        vtk = ng.VTKOutput(
            ma=self.mesh,
            coefs=[gfu],
            names=["q(T)"],
            filename=filename,
            subdivision=3,
        )
        vtk.Do()

    # Solver ------------------------------------------------------------------
    def solve(self, mu, t):
        """Solve the problem for a given parameter vector."""
        dt = t[1] - t[0]
        MV = sum(1 / mui * MVi for mui, MVi in zip(mu, self.MVs))
        MV_A = spla.spsolve(MV, self.S)

        D = self.MWinv @ self.S.T @ MV_A
        D *= dt / 2
        D += 2 / dt * sp.identity(D.shape[0], format="csr")
        D_factor = spla.factorized(D)

        Q = np.zeros((self.Nx, len(t)))
        P = np.zeros((self.Nx, len(t)))

        Q[:, 0] = self.q0
        P[:, 0] = self.p0

        # Symplectic time integrator
        for i in range(1, len(t)):
            qHalf = D_factor(2 / dt * Q[:, i - 1] + P[:, i - 1])
            Q[:, i] = 2 * qHalf - Q[:, i - 1]
            P[:, i] = 4 / dt * (qHalf - Q[:, i - 1]) - P[:, i - 1]

        return Q, P

    def solve_multi(self, muarr, t):
        """Solve the problem for multiple parameter vectors."""
        Q_list, P_list = [], []

        for mu in muarr:
            Q, P = self.solve(mu, t)
            Q_list.append(Q)
            P_list.append(P)

        return np.array(Q_list), np.array(P_list)

    # Intrusive reduced-order modeling ----------------------------------------
    def solveROM(
        self,
        basis: opinf.basis.PODBasis,
        mu,
        t,
        check_is_Morthonormal: bool = False,
    ):
        """Constructs the Galerkin ROM and returns the corresponding solution
        for a given parameter vector.

        Given a basis matrix U satsifying U^T M U = I, the Galerkin ROM is

            [ dq'/dt ]   [  0  I ] [ A'(μ)  0 ] [ q' ]
            [ dp'/dt ] = [ -I  0 ] [   0    I ] [ p' ]

        where q = Uq' and A'(μ) = U^T A(μ) U.

        Parameters
        ----------
        basis : opinf.basis.PODBasis
            POD basis, already fit to training data.
        check_is_Morthonormal : bool
            If ``True``, verify that U^T M U = I.
        """
        U = basis.entries
        if check_is_Morthonormal:
            if not np.allclose((U.T @ self.M) @ U, np.eye(U.shape[1])):
                raise ValueError("basis is not M-orthonormal")

        dt = t[1] - t[0]
        MV = sum(1 / mui * MVi for mui, MVi in zip(mu, self.MVs))
        MV_A = spla.spsolve(MV, self.S)
        D = self.MWinv @ self.S.T @ MV_A
        Dh = basis.compress(D.toarray() @ basis.entries)
        Dh *= dt / 2
        Dh += 2 / dt * sp.identity(Dh.shape[0])
        rhsM = inv(Dh)

        Q = np.zeros((self.Nx, len(t)))
        P = np.zeros((self.Nx, len(t)))

        Q[:, 0] = basis.decompress(basis.compress(self.q0))
        P[:, 0] = basis.decompress(basis.compress(self.p0))

        for i in range(1, len(t)):
            q0, p0 = basis.compress(Q[:, i - 1]), basis.compress(P[:, i - 1])
            qHalf = rhsM @ (2 / dt * q0 + p0)
            Q[:, i] = basis.decompress(2 * qHalf - q0)
            P[:, i] = basis.decompress(4 / dt * (qHalf - q0) - p0)

        return Q, P

    def solve_multiROM(self, basis, muarr, t):
        """Solve the problem with the Galerkin ROM for multiple parameter
        vectors.
        """
        Q_list, P_list = [], []
        for mu in muarr:
            Q, P = self.solveROM(basis, mu, t)
            Q_list.append(Q)
            P_list.append(P)

        return np.array(Q_list), np.array(P_list)

    def Hamiltonian(self, Q, P, mu):
        """Compute the time-dependent Hamiltonian given position snapshots Q
        and momentum snapsots P corresponding to the parameter vector mu.
        """
        H = np.empty(Q.shape[1])
        MV = sum(1 / mui * MVi for mui, MVi in zip(mu, self.MVs))
        MVinv = self.S.T @ spla.inv(MV) @ self.S

        for i in range(Q.shape[1]):
            H[i] = (
                P[:, i].T @ (self.MW @ P[:, i]) + Q[:, i].T @ MVinv @ Q[:, i]
            )

        return 0.5 * H

    def reduced_Hamiltonian(self, basis: opinf.basis.PODBasis, Q, P, mu):
        """Compute the time-dependent reduced Hamiltonian given
        (high- dimensional) position snapshots Q and momentum snapsots P
        corresponding to the parameter vector mu and the POD basis already fit
        to training data.
        """
        H = np.empty(Q.shape[1])
        MV = sum(1 / mui * MVi for mui, MVi in zip(mu, self.MVs))
        MVinvh = (
            basis.entries.T @ self.S.T @ spla.inv(MV) @ self.S @ basis.entries
        )

        Qh = basis.compress(Q)
        Ph = basis.compress(P)

        for i in range(Q.shape[1]):
            H[i] = Ph[:, i].T @ Ph[:, i] + Qh[:, i].T @ MVinvh @ Qh[:, i]
        return 0.5 * H

    def animate(self, Q, skip=5):
        """Animate a single evolution profile in time in Jupyter notebook.

        Parameters
        ----------
        Q : (Nx, Nt) ndarray
            Trajectory to animate.
        skip : int
            Animate every `skip` snapshots, so the total number of
            frames is `Nt // skip`.
        """
        if Q.ndim != 2:
            raise ValueError("two-dimensional data required for animation")
        x = self.nodes

        # Initialize the figure and subplots.
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 4), dpi=200)
        lines = [ax.plot([], [])[0]]

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(index):
            lines[0].set_data(x, Q[:, index * skip])
            ax.set_title(rf"$t = t_{{{index*skip}}}$")
            return lines

        ax.axvline(self.x1, linestyle=":", linewidth=0.5)
        ax.axvline(self.x2, linestyle=":", linewidth=0.5)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(Q.min() * 0.95, Q.max() * 1.05)
        ax.set_title(r"$t = t_{0}$")

        a = matplotlib.animation.FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=Q.shape[1] // skip,
            interval=50,
            blit=True,
        )
        plt.close(fig)
        return IPython.display.HTML(a.to_jshtml())


class WaveFEM1D(WaveFEM):
    r"""Finite element model for the one-dimensional wave equation with a
    piecewise constant wave speed and homogeneous Dirichlet
    boundary conditions.

    The governing equation is

        d^2q/dt^2 = d/dx[c^2 dq/dx],

    defined over the one-dimensional spatial domain [0, L], with boundary
    conditions q(x,t) = 0 and initial conditions

        q(x,0) = e^(-(x - L/2)^2) sin(x), dq/dt (x,0) = 0.

    The semi-discrete system has the block form

        [ M_W   0  ] [ dq/dt ]   [  0  I ] [ A(μ)   0  ] [ q ]
        [  0   M_W ] [ dp/dt ] = [ -I  0 ] [  0    M_W ] [ p ],

    where A(μ) has an affine parametric decomposition

        A(μ) = 1/μ_1^2 A_1 + ... + 1/μ_p^2 A_p \approx S^T M_V(μ)^{-1} S.

    Parameters
    ----------
    L : float
        Length of the spatial domain.
    num_elements : int
        Number of mesh elements.
    orderW : int
        Polynomial order of the L2 finite element space.
    orderV : int
        Polynomial order of the HDiv finite element space.
    """

    dim = 1
    parameter_dimension = 4

    def __init__(
        self,
        L: float = 2 * np.pi,
        num_elements: int = 500,
        orderW: int = 0,
        orderV: int = 1,
    ):
        self.L, self.ne = L, num_elements
        self.x1, self.x2, self.x3 = L / 4, L / 2, 3 * L / 4
        super().__init__(orderW, orderV)

    # Implement abstract methods ----------------------------------------------
    def create_mesh(self):
        """Create a 1D spatial mesh over [0, L] with ``ne`` elements."""
        return Make1DMesh(self.ne, mapping=lambda x: self.L * x)

    def initial_condition(self):
        return ng.exp(-((ng.x - (self.L / 2)) ** 2)) * ng.sin(ng.x), 0

    def getS(self):
        S = ng.BilinearForm(trialspace=self.W, testspace=self.V)
        S += self.W.TrialFunction() * ng.grad(self.V.TestFunction()) * ng.dx
        return S

    def getMVs(self) -> list:
        u, v = self.V.TnT()
        ind0 = ng.IfPos(self.x1 - ng.x, 1, 0)
        ind1 = ng.IfPos(self.x2 - ng.x, 1, 0) * ng.IfPos(ng.x - self.x1, 1, 0)
        ind2 = ng.IfPos(self.x3 - ng.x, 1, 0) * ng.IfPos(ng.x - self.x2, 1, 0)
        ind3 = ng.IfPos(ng.x - self.x3, 1, 0)

        out = []
        for ind in (ind0, ind1, ind2, ind3):
            Ai = ng.BilinearForm(self.V)
            Ai += ind * u * v * ng.dx
            out.append(self._tocoo(Ai).tocsc())
        return out

    # Visualization -----------------------------------------------------------
    def plot(self, Q, indices=(0, 10, 50, 100, 250, 500, 900)):
        """Plot snapshots of a solution trajectory over the spatial domain.

        Parameters
        ----------
        Q : (Nx, Nt) ndarray
            Trajectory to plot.
        indices : tuple
            Time indices to plot.
        """
        x = self.nodes

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(indices)))
        for j, c in zip(indices, colors):
            if j < Q.shape[1]:
                ax.plot(x, Q[:, j], color=c, label=f"$t = t_{{{j}}}$")
        ax.axvline(self.x1, linestyle=":", linewidth=0.5)
        ax.axvline(self.x2, linestyle=":", linewidth=0.5)
        ax.axvline(self.x3, linestyle=":", linewidth=0.5)

        fig.subplots_adjust(right=0.85)
        ax.legend(
            loc="center right",
            bbox_to_anchor=(1, 0.5),
            bbox_transform=fig.transFigure,
        )
        return fig, ax


class WaveFEM2D(WaveFEM):
    r"""Finite element model for the two-dimensional wave equation with a
    piecewise constant wave speed and homogeneous Dirichlet boundary
    conditions.

    The governing equation is

        d^2q/dt^2 = div[c^2 grad(q)],

    defined over the two-dimensional spatial domain [0, L] x [0, L], with
    boundary conditions q(x,t) = 0 and initial conditions

        q(x,0) = e^(-(x1 - L/2)^2 - (x2 - L/2)^2) sin(x1/2) sin(x2/2).

    where x = (x1, x2). The semi-discrete system has the block form

        [ M_W   0  ] [ dq/dt ]   [  0  I ] [ A(μ)   0  ] [ q ]
        [  0   M_W ] [ dp/dt ] = [ -I  0 ] [  0    M_W ] [ p ],

    where A(μ) has an affine parametric decomposition

        A(μ) = 1/μ_1^2 A_1 + ... + 1/μ_p^2 A_p \approx S^T M_V(μ)^{-1} S.

    Parameters
    ----------
    L : float
        Length of one side of the spatial domain.
    h : float
        Maximum mesh spacing.
    orderW : int
        Polynomial order of the L2 finite element space.
    orderV : int
        Polynomial order of the HDiv finite element space.
    """

    dim = 2
    parameter_dimension = 4
    plot_settings = {
        "camera": {
            "transformations": [
                {"type": "rotateX", "angle": -90},
            ]
        },
        "deformation": 3.0,
        "edges": False,
        "mesh": False,
    }

    def __init__(
        self,
        L: float = 2 * np.pi,
        h: float = 0.25,
        orderW: int = 1,
        orderV: int = 1,
    ):
        self.L, self.h = L, h
        super().__init__(orderW, orderV)

    # Implement abstract methods ----------------------------------------------
    def create_mesh(self):
        """Create a 2D spatial mesh over [0, L] x [0, L] with max spacing h."""
        L = self.L
        L2 = L / 2

        geo = SplineGeometry()
        geo.AddRectangle(
            (0, 0),
            (L2, L2),
            bcs=["b1", "r1", "t1", "l1"],
            leftdomain=1,
        )
        geo.AddRectangle(
            (L2, 0),
            (L, L2),
            bcs=["b2", "r2", "t2", "l2"],
            leftdomain=2,
        )
        geo.AddRectangle(
            (0, L2),
            (L2, L),
            bcs=["b3", "r3", "t3", "l3"],
            leftdomain=3,
        )
        geo.AddRectangle(
            (L2, L2),
            (L, L),
            bcs=["b4", "r4", "t4", "l4"],
            leftdomain=4,
        )

        geo.SetMaterial(1, "d1")
        geo.SetMaterial(2, "d2")
        geo.SetMaterial(3, "d3")
        geo.SetMaterial(4, "d4")

        return ng.Mesh(geo.GenerateMesh(maxh=self.h))

    def initial_condition(self):
        mid = self.L / 2
        return (
            ng.exp(-0.01 * (ng.x - mid) ** 2 - 0.01 * (ng.y - mid) ** 2)
            * ng.sin(ng.x / 2)
            * ng.sin(ng.y / 2)
        ), 0

    def getS(self):
        S = ng.BilinearForm(trialspace=self.W, testspace=self.V)
        S += self.W.TrialFunction() * ng.div(self.V.TestFunction()) * ng.dx
        return S

    def getMVs(self) -> list:
        u, v = self.V.TnT()
        materials = self.mesh.GetMaterials()
        out = []

        for i in range(len(materials)):
            Mi = ng.BilinearForm(self.V, check_unused=False)
            Mi += u * v * ng.dx(definedon=self.mesh.Materials(materials[i]))
            out.append(self._tocoo(Mi).tocsc())
        return out

    def plot(self, Q):
        """Plot snapshots of a solution trajectory over the spatial domain.

        Parameters
        ----------
        Q : (Nx, Nt) ndarray
            Trajectory to plot.
        indices : tuple
            Time indices to plot.
        """

        gfu = ng.GridFunction(self.W)
        gfu.vec.FV().NumPy()[:] = Q[:, 0]
        scene = Draw(gfu, settings=self.plot_settings)

        for j in range(1, Q.shape[1]):
            print(f"t = t_{j}", end="\r")
            gfu.vec.FV().NumPy()[:] = Q[:, j]
            scene.Redraw()
