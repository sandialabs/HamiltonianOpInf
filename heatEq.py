# heatEq.py
"""Models for the heat equation with piecewise constant diffusion."""

__all__ = [
    "HeatFEM",
    "HeatFEM1D",
    "HeatFEM2D",
]

import abc
import opinf
import numpy as np
import IPython.display
import scipy.integrate
import scipy.sparse as sp
import matplotlib.animation
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla

import ngsolve as ng
from ngsolve.webgui import Draw
from ngsolve.meshes import Make1DMesh
from netgen.geom2d import SplineGeometry

import models


class HeatFEM(abc.ABC):
    """Base class for finite element models of the heat equation with a
    piecewise constant diffusion coefficient and homogeneous Dirichlet
    boundary conditions. The semi-discrete system has the form

        M dq/dt = -S(mu)q(t)

    Where S(mu) has an affine parametric decomposition

        -S(mu) = mu_1 S_1 + ... + mu_p S_p.

    The initial condition and the underlying spatial dimension and geometry
    are specified in child classes.

    Parameters
    ----------
    t0 : float
        Initial time.
    tf : float
        Final time.
    nt : int
        Number of time steps.
    order : int
        Polynomial order of the finite element space.
    """

    dim = NotImplemented
    dirichlet_BCs = NotImplemented
    parameter_dimension = NotImplemented

    def __init__(self, order: int = 1):
        # Initialize the finite element mesh and space.
        self.mesh = self.create_mesh()
        self.V = ng.H1(self.mesh, order=order, dirichlet=self.dirichlet_BCs)
        self.__x = np.array([self.mesh[v].point for v in self.mesh.vertices])
        if self.parameter_dimension == 1:
            self.__x = np.ravel(self.__x)

        # Assemble the mass matrix.
        u, v = self.V.TnT()
        M = ng.BilinearForm(self.V)
        M += u * v * ng.dx
        M = self._tocoo(M).tocsc()

        # Assemble stiffness matrices and the initial condition.
        As = self.stiffness_matrices()
        q0 = self._asarray(self.initial_condition())

        # Get the degrees of freedom not accounted for by boundary conditions.
        self.free = self.V.FreeDofs()
        self.M = M[self.free][:, self.free]
        self.Minv = spla.splu(self.M).solve
        self.As = [A[self.free][:, self.free] for A in As]
        self.q0 = q0[self.free]

    @property
    def Nx(self) -> int:
        """Size of the spatial discretization."""
        return self.V.ndof

    @property
    def Nx_free(self) -> int:
        """Number of degrees of freedom not fixed by boundary conditions."""
        return self.q0.size

    @property
    def nodes(self) -> np.ndarray:
        """Nodes in the spatial mesh."""
        return self.__x

    def __str__(self):
        """String representation."""
        interval = f"[0, {self.L:.4f}]"
        domain = " x ".join([interval] * self.dim)
        return "\n".join(
            [
                "Parametric heat equation finite element model",
                f"  Spatial domain ({self.dim:d}D): {domain}",
                f"  Discretization size: {self.Nx}",
                f"  Degrees of freedom:  {self.Nx_free}",
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
    def stiffness_matrices(self) -> list:
        """Assemble the (parameter-independent) matrices S_1, ..., S_p
        where the stiffness matrix is -S(mu) = mu_1 S_1 + ... + mu_p S_p.
        """
        raise NotImplementedError

    # Utilities ---------------------------------------------------------------
    def _tocoo(self, A) -> sp.coo_array:
        """Convert an ngsolve matrix to a SciPy sparse COO array."""
        coo = A.Assemble().mat.COO()
        return sp.coo_array((coo[2], coo[:2])).tocsc()

    def _asarray(self, func):
        """Convert an ngsolve function to a NumPy array."""
        f = ng.GridFunction(self.V)
        f.Set(func)
        return f.vec.FV().NumPy()

    def pad(self, q_free: np.ndarray) -> np.ndarray:
        """Pad with the homogeneous boundary conditions.

        Parameters
        ----------
        q_free : (N_free, ...) ndarray
            Vector of coefficients for the degrees of freedom.

        Returns
        -------
        q_all : (Nx, ....) ndarray
            Values over the nodes, including the zero boundary conditions.
        """
        shape = list(q_free.shape)
        shape[0] = self.Nx
        q = np.zeros(shape)
        q[self.free] = q_free[:]
        return q

    def sample_parameters(
        self,
        low: float,
        high: float,
        num_samples: int = 10,
        train_ratio: float = 0.8,
        randseed: int = 0,
    ):
        """Sample the parameter space log-uniformly to generate training and
        testing sets.

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
        samples = 10 ** np.random.uniform(
            low=np.log10(low),
            high=np.log10(high),
            size=(num_samples, self.parameter_dimension),
        )

        # Split into training and testing sets.
        split = int(train_ratio * num_samples)
        return samples[:split], samples[split:]

    # Solver ------------------------------------------------------------------
    def derivative(self, mu, q):
        """Evaluate the time derivative dq/dt = -M^{-1}S(mu)q."""
        A = sum(mui * Ai for mui, Ai in zip(mu, self.As))
        return self.Minv(A @ q)

    def solve(self, mu, t):
        """Solve the problem for a given parameter vector."""
        A = sum(mui * Ai for mui, Ai in zip(mu, self.As))
        jac = self.Minv(A.toarray())

        def fun(tt, y):
            return self.Minv(A @ y)

        # Solve over the non-boundary degrees of freedom.
        return scipy.integrate.solve_ivp(
            fun=fun,
            t_span=[t[0], t[-1]],
            y0=self.q0,
            method="BDF",
            t_eval=t,
            jac=jac,
            vectorized=True,
        ).y

    def solve_multi(self, muarr, t):
        """Solve the problem for multiple parameter vectors."""
        return np.array([self.solve(mu, t) for mu in muarr])

    # Intrusive reduced-order modeling ----------------------------------------
    def construct_intrusive_ROM(
        self,
        basis: opinf.basis.PODBasis,
        check_is_Morthonormal: bool = False,
    ) -> opinf.ParametricROM:
        """Construct a Galerkin ROM for this model via intrusive projection.

        Given a basis matrix U satsifying U^T M U = I, the Galerkin ROM is

            dq'/dt = -U^T S(mu) U q'(t)

        where q = U q'.

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

        model = models.TensorContinuousModel(self.parameter_dimension)
        Ahats = [(U.T @ A) @ U for A in self.As]
        model.operators[0].set_entries(Ahats)
        model.state_dimension = U.shape[1]

        return opinf.ParametricROM(basis=basis, model=model)


class HeatFEM1D(HeatFEM):
    """Finite element model for the one-dimensional heat equation with a
    piecewise constant diffusion coefficient and homogeneous Dirichlet
    boundary conditions.

    The governing equation is

        dq/dt = d/dx[c dq/dx],

    defined over the one-dimensional spatial domain [0, L], with boundary
    conditions q(x,t) = 0 and initial condition

        q(x,0) = e^(-(x - L/2)^2) sin(x/2).

    The semi-discrete system has the form

        M dq/dt = -S(mu)q(t),

    where S(mu) has an affine parametric decomposition

        -S(mu) = mu_1 S_1 + ... + mu_p S_p.

    Parameters
    ----------
    L : float
        Length of the spatial domain.
    num_elements : int
        Number of mesh elements.
    order : int
        Polynomial order of the finite element space.
    """

    dim = 1
    dirichlet_BCs = "left|right"
    parameter_dimension = 3

    def __init__(
        self,
        L: float = 2 * np.pi,
        num_elements: int = 500,
        order: int = 1,
    ):
        self.L, self.ne = L, num_elements
        self.x1, self.x2 = L / 3, 2 * L / 3
        super().__init__(order)

    # Implement abstract methods ----------------------------------------------
    def create_mesh(self):
        """Create a 1D spatial mesh over [0, L] with ``ne`` elements."""
        return Make1DMesh(self.ne, mapping=lambda x: self.L * x)

    def initial_condition(self):
        return ng.exp(-((ng.x - (self.L / 2)) ** 2)) * ng.sin(ng.x / 2)

    def stiffness_matrices(self) -> list:
        u, v = self.V.TnT()
        ind0 = ng.IfPos(self.x1 - ng.x, 1, 0)
        ind1 = ng.IfPos(self.x2 - ng.x, 1, 0) * ng.IfPos(ng.x - self.x1, 1, 0)
        ind2 = ng.IfPos(ng.x - self.x2, 1, 0)

        out = []
        for ind in (ind0, ind1, ind2):
            Ai = ng.BilinearForm(self.V)
            Ai += ind * ng.grad(u) * ng.grad(v) * ng.dx
            out.append(-self._tocoo(Ai).tocsr())
        return out

    # Visualization -----------------------------------------------------------
    def plot(self, Qfree, indices=(0, 10, 50, 100, 250, 500, 900)):
        """Plot snapshots of a solution trajectory over the spatial domain.

        Parameters
        ----------
        Qfree : (Nx_free, Nt) ndarray
            Trajectory to plot.
        indices : tuple
            Time indices to plot.
        """
        Q = self.pad(Qfree)
        x = self.nodes

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(indices)))
        for j, c in zip(indices, colors):
            if j < Q.shape[1]:
                ax.plot(x, Q[:, j], color=c, label=f"$t = t_{{{j}}}$")
        ax.axvline(self.x1, linestyle=":", linewidth=0.5)
        ax.axvline(self.x2, linestyle=":", linewidth=0.5)

        ax.set_xlim(x.min(), x.max())
        fig.subplots_adjust(right=0.85)
        ax.legend(
            loc="center right",
            bbox_to_anchor=(1, 0.5),
            bbox_transform=fig.transFigure,
        )
        return fig, ax

    def animate(self, Qfree, skip=5):
        """Animate a single evolution profile in time in Jupyter notebook.

        Parameters
        ----------
        Qfree : (Nx_free, Nt) ndarray
            Trajectory to animate.
        skip : int
            Animate every `skip` snapshots, so the total number of
            frames is `Nt // skip`.
        """
        if Qfree.ndim != 2:
            raise ValueError("two-dimensional data required for animation")
        Q = self.pad(Qfree)
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


class HeatFEM2D(HeatFEM):
    """Finite element model for the one-dimensional heat equation with a
    piecewise constant diffusion coefficient and homogeneous Dirichlet
    boundary conditions.

    The governing equation is

        dq/dt = div(c grad(q)],

    defined over the square, two-dimensional spatial domain [0, L] x [0, L],
    with boundary conditions q(x,t) = 0 and initial condition

        q(x,0) = e^(-(x1 - L/2)^2 - (x2 - L/2)^2) sin(x1/2) sin(x2/2)

    where x = (x1, x2). The semi-discrete system has the form

        M dq/dt = -S(mu)q(t),

    where S(mu) has an affine parametric decomposition

        -S(mu) = mu_1 S_1 + ... + mu_p S_p.

    Parameters
    ----------
    L : float
        Length of one side of the spatial domain.
    h : float
        Maximum mesh spacing.
    order : int
        Polynomial order of the finite element space.
    """

    dim = 2
    dirichlet_BCs = "b1|b2|l1|l3|t3|t4|r2|r4"
    parameter_dimension = 4
    plot_settings = {
        "camera": {
            "transformations": [
                {"type": "rotateY", "angle": 20},
                {"type": "rotateZ", "angle": 40},
            ]
        },
        "deformation": 3.0,
        "edges": False,
        "mesh": False,
    }

    def __init__(self, L: float = 2 * np.pi, h: float = 0.25, order: int = 1):
        self.L, self.h = L, h
        super().__init__(order)

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
            ng.exp(-((ng.x - mid) ** 2) - ((ng.y - mid) ** 2))
            * ng.sin(ng.x / 2)
            * ng.sin(ng.y / 2)
        )

    def stiffness_matrices(self):
        u, v = self.V.TnT()
        materials = self.mesh.GetMaterials()

        out = []
        for i in range(len(materials)):
            Ai = ng.BilinearForm(self.V, check_unused=False)
            Ai += (
                ng.grad(u)
                * ng.grad(v)
                * ng.dx(definedon=self.mesh.Materials(materials[i]))
            )
            out.append(-self._tocoo(Ai).tocsr())
        return out

    # Visualization -----------------------------------------------------------
    def plot(self, Qfree):
        """Plot snapshots of a solution trajectory over the spatial domain.
        In a notebook, this is displayed as a once-through animation.

        Parameters
        ----------
        Qfree : (Nx_free, Nt) ndarray
            Trajectory to plot.
        indices : tuple
            Time indices to plot.
        """
        Q = self.pad(Qfree)
        gfu = ng.GridFunction(self.V)
        gfu.vec.FV().NumPy()[:] = Q[:, 0]
        scene = Draw(gfu, settings=self.plot_settings)

        for j in range(1, Q.shape[1]):
            print(f"t = t_{j}", end="\r")
            gfu.vec.FV().NumPy()[:] = Q[:, j]
            scene.Redraw()
