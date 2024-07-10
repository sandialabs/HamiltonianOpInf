# Convenience functions for running the ROMs
# Anthony Gruber 3-31-2023

# Standared numpy/scipy imports
import numpy as np
from numpy.linalg import norm, eigvals, solve
from scipy.linalg import lu_factor, lu_solve, det
from scipy.sparse import csc_matrix, identity, issparse
from scipy.sparse.linalg import factorized, spsolve

# For additional plotting and gifing
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn.utils.extmath import randomized_svd

import scipy.sparse as sparse
import sys


class Linear_Variational_ROM:
    def __init__(self, snapshots):
        self.N, self.N_t = snapshots.shape
        self.snapshots   = snapshots

    def set_reduced_basis(self, option, centered=False,
                          randomized=False, rmax=120):
        if not randomized:
            svdalg = partial(np.linalg.svd, full_matrices=False)
        else:
            svdalg = partial(randomized_svd, n_components=rmax)

        if centered:
            X = self.snapshots - self.snapshots[:,0].reshape(-1,1)
            self.centered = True
        else:
            X = self.snapshots
            self.centered = False

        if option == 'POD':
            UU, SS = svdalg(X)[:2]
            self.reduced_basis, self.basis_evals = UU[:,:rmax], SS[:rmax]
        if option == 'cotangent_lift':
            half_N      = self.N // 2
            X_long      = np.concatenate((X[:half_N], X[half_N:]), axis=1)
            U_block, SS = svdalg(X_long)[:2]
            self.reduced_basis = [U_block[:,:rmax], U_block[:,:rmax]]
            self.basis_evals   = [SS[:rmax], SS[:rmax]]
        if option == 'block_qp':
            half_N   = self.N // 2
            U_q, S_q = np.linalg.svd(X[:half_N])[:2]
            U_p, S_p = np.linalg.svd(X[half_N:])[:2]
            self.reduced_basis = [U_q[:,:rmax], U_p[:,:rmax]]
            self.basis_evals   = [S_q[:rmax], S_p[:rmax]]
        if option == 'block_qq':
            half_N   = self.N // 2
            U_q, S_q = np.linalg.svd(X[:half_N])[:2]
            self.reduced_basis = [U_q[:,:rmax], U_q[:,:rmax]]
            self.basis_evals   = [S_q[:rmax], S_q[:rmax]]
        if option == 'complex_SVD':  # Will need to build mtx later
            half_N   = self.N // 2
            U_c, S_c = np.linalg.svd(X[:half_N] + X[half_N:]*1j)[:2]
            self.reduced_basis = [U_c[:,:rmax], U_c[:,:rmax]]
            self.basis_evals   = [S_c[:rmax], S_c[:rmax]]

    def compute_basis_energies(self):
        try: self.reduced_basis, self.basis_evals
        except: 
            print('need to set reduced basis first!')
            exit
        if not isinstance(self.reduced_basis, list):
            eigs = self.basis_evals
            self.basis_energies = np.cumsum(eigs / np.sum(eigs))*100
        else:
            eigsList = self.basis_evals
            self.basis_energies = [np.cumsum(eigs / np.sum(eigs))*100 
                                   for eigs in eigsList]

    def encode(self, x, r):
        N = x.shape[0]
        if not isinstance(self.reduced_basis, list):
            RB = self.reduced_basis[:,:r]
        else:
            RB = self.basis_from_list(self.reduced_basis, r)
        if self.centered:
            return RB.T @ (x - self.snapshots[:,0].reshape(-1,1))
        else:
            return RB.T @ x      

    def decode(self, x_hat):
        r = x_hat.shape[0]
        if not isinstance(self.reduced_basis, list):
            RB = self.reduced_basis[:,:r]
        else:
            RB = self.basis_from_list(self.reduced_basis, r)
        if self.centered:
            return self.snapshots[:,0].reshape(-1,1) + RB @ x_hat
        else:
            return RB @ x_hat
        
    def project(self, x, r):
        return self.decode(self.encode(x, r))

    def basis_from_list(self, basis_list, two_r):
        r    = two_r // 2
        if not np.iscomplexobj(basis_list[0]):
            U_11 = basis_list[0][:,:r]
            U_22 = basis_list[1][:,:r]
            ZZ   = np.zeros_like(U_11)
            return np.block([[U_11, ZZ], [ZZ, U_22]])
        else:
            Ur = np.real(basis_list[0])[:,:r]
            Ui = np.imag(basis_list[0])[:,:r]
            return np.block([[Ur, -Ui], [Ui, Ur]])


class Linear_Hamiltonian_ROM(Linear_Variational_ROM):
    def __init__(self, snapshots):
        super(Linear_Hamiltonian_ROM, self).__init__(snapshots)

    def assemble_naive_ROM(self, r, J, A):
        try: self.reduced_basis
        except: 
            print('need to set reduced basis first!')
            exit

        x_0 = self.snapshots[:,0]

        if not isinstance(self.reduced_basis, list):
            RB = self.reduced_basis[:,:r]
        else:
            # list bases have to recompute everything
            RB = self.basis_from_list(self.reduced_basis, r)

        self.JA_hat = RB.T @ J @ A @ RB
        self.ic     = RB.T @ x_0
        self.x0b4JA = np.zeros(r)
        if self.centered:
            self.ic     = np.zeros(r)
            self.x0b4JA = RB.T @ J @ A @ x_0

    def assemble_Hamiltonian_ROM(self, r, J, A):
        try: self.reduced_basis
        except: 
            print('need to set reduced basis first!')
            exit

        x_0 = self.snapshots[:,0]

        if not isinstance(self.reduced_basis, list):
            RB = self.reduced_basis[:,:r]
        else:
            # list bases have to recompute everything
            RB = self.basis_from_list(self.reduced_basis, r)
        
        self.J_hat = RB.T @ J @ RB
        self.A_hat = RB.T @ A @ RB
        self.ic    = RB.T @ x_0
        self.x0b4A = np.zeros(r)
        if self.centered:
            self.x0b4A = RB.T @ A @ x_0
            self.ic    = np.zeros(r)

    def integrate_naive_ROM(self, times):
        try: JA_hat = self.JA_hat
        except:
            print('naive ROM operators are not set!')
        
        # Fix sparse/dense errors
        if issparse(JA_hat):
            JA_hat = JA_hat.todense()    
    
        dt = times[1] - times[0]
        r  = len(self.ic)
        
        x_hat = np.zeros((r, times.shape[0]))
        x_hat[:,0] = self.ic

        LHS = np.eye(r) - dt/2 * JA_hat
        lu, piv = lu_factor(LHS)

        for i,t in enumerate(times[:-1]):
            rhs          = dt * (self.x0b4JA + JA_hat @ x_hat[:,i])
            del_x_hat_i  = lu_solve((lu, piv), rhs)
            x_hat[:,i+1] = x_hat[:,i] + del_x_hat_i

        self.x_hat = x_hat

    def integrate_Hamiltonian_ROM(self, times, eps=0., inconsistent=False):
        try: J_hat, A_hat = self.J_hat, self.A_hat
        except:
            print('Hamiltonian ROM operators are not set!')

        # Fix sparse/dense errors
        if issparse(J_hat) or issparse(A_hat):
            J_hat = J_hat.todense()
            A_hat = A_hat.todense()

        dt = times[1] - times[0]
        r  = len(self.ic)
        if eps != 0.:
            n = r // 2
            J_small = np.block([[np.zeros((n,n)), np.eye(n)],
                                [-np.eye(n), np.zeros((n,n))]])
            J_hat += eps * J_small
        
        x_hat = np.zeros((r, times.shape[0]))
        x_hat[:,0] = self.ic

        if inconsistent:
            J_hat_A_hat = J_hat @ A_hat
            LHS         = np.eye(r) - dt/2 * J_hat_A_hat
            rhs_func    = lambda x: dt * (J_hat @ self.x0b4A + J_hat_A_hat @ x)
        else:
            LHS      = J_hat + dt/2 * A_hat
            rhs_func = lambda x: -dt * (self.x0b4A + A_hat @ x)

        lu, piv = lu_factor(LHS)

        for i,t in enumerate(times[:-1]):
            rhs          = rhs_func(x_hat[:,i])
            del_x_hat_i  = lu_solve((lu, piv), rhs)
            x_hat[:,i+1] = x_hat[:,i] + del_x_hat_i

        self.x_hat = x_hat

    # Solving generic OpInf Problem with Willcox method.
    def infer_generic(self, r, X, Xt, eps=0., reproject=False):
        try: self.reduced_basis
        except: 
            print('need to set reduced basis first!')
            exit

        x_0 = self.snapshots[:,0]

        if self.centered:
            X = X - X[:,0].reshape(-1,1)

        if not isinstance(self.reduced_basis, list):
            RB = self.reduced_basis[:,:r]
        else:
            # list bases have to recompute everything
            RB = self.basis_from_list(self.reduced_basis, r)

        X_hat_X_hat_T = RB.T @ X @ X.T @ RB
        Xt_hat        = RB.T @ Xt
        self.ic       = RB.T @ x_0
        self.x0b4JA   = np.zeros(r)
        if self.centered:
            self.x0b4JA = RB.T @ Xt[:,0]
            Xt_hat     -= self.x0b4JA.reshape(-1,1)
            self.ic     = np.zeros(r)

        if reproject:
            Xt_hat  = RB.T @ self.J @ self.A @ RB @ RB.T @ X

        rhs         = Xt_hat @ X.T @ RB
        LHS         = X_hat_X_hat_T + eps * np.eye(r)
        self.JA_hat = solve(LHS, rhs.T).T

    def infer_canonical_Hamiltonian(self, r, X, Xt, J, eps=0., old=False, 
                                    reproject=False):
        try: self.reduced_basis
        except: 
            print('need to set reduced basis first!')
            exit

        x_0 = self.snapshots[:,0]

        if self.centered:
           X = X - X[:,0].reshape(-1,1)

        if not isinstance(self.reduced_basis, list):
            RB = self.reduced_basis[:,:r]
        else:
            # list bases have to recompute everything
            RB = self.basis_from_list(self.reduced_basis, r)
            
        X_hat_X_hat_T  = RB.T @ X @ X.T @ RB
        Xt_hat         = RB.T @ Xt
        Xt_part        = RB.T @ -J @ Xt
        X_hat          = RB.T @ X
        self.J_hat     = RB.T @ J @ RB
        self.ic        = RB.T @ x_0
        self.x0b4A     = np.zeros(r)
        if self.centered:
            self.x0b4A = RB.T @ -J @ Xt[:,0]
            self.ic    = np.zeros(r)

        if reproject:
            Xt_part = RB.T @ self.A @ RB @ RB.T @ X
            Xt_hat = Xt_part

        if eps != 0.:
            n = r // 2
            J_small = np.block([[np.zeros((n,n)), np.eye(n)],
                                [-np.eye(n), np.zeros((n,n))]])
        else:
            J_small = np.zeros((r,r))
            
        # J_hat = self.J_hat + eps * J_small
        J_hat = self.J_hat

        if old:  # Not compatible with reprojection
            temp   = (J_hat.T @ Xt_hat - self.x0b4A.reshape(-1,1)) @ X_hat.T
        else:
            temp = (Xt_part - self.x0b4A.reshape(-1,1)) @ X_hat.T
            # temp   = (J_hat.T @ Xt_hat - self.x0b4A.reshape(-1,1)) @ X_hat.T
        if reproject:
            temp   = Xt_part @ X_hat.T
            # temp   = Xt_hat @ X.T @ RB

        rhs        = temp + temp.T
        P          = csc_matrix(np.kron(np.eye(r), X_hat_X_hat_T) 
                                + np.kron(X_hat_X_hat_T, np.eye(r)))
        reg        = 2 * eps * identity(r**2)

        A_hat      = spsolve(P+reg, vec(rhs).squeeze()).reshape((r,r), order='F')
        self.A_hat = 0.5 * (A_hat + A_hat.T)

        # rhs         = Xt_hat @ X.T @ RB
        # LHS         = X_hat_X_hat_T
        # self.A_hat = solve(LHS, rhs.T).T

    def infer_noncanonical_Hamiltonian(self, r, Xt, gradH, A, eps=0.):
        try: self.reduced_basis
        except: 
            print('need to set reduced basis first!')
            exit     

        x_0 = self.snapshots[:,0]

        if not isinstance(self.reduced_basis, list):
            # check if operators have been precomputed
            try: (self.big_gradH_hat_gradH_hat_T, self.big_Xt_hat_gradH_hat_T,
                  self.big_A_hat, self.big_ic, self.big_x0b4A)
            except:
                # if not, compute them and store
                RB                             = self.reduced_basis
                self.big_gradH_hat_gradH_hat_T = RB.T @ gradH @ gradH.T @ RB
                self.big_Xt_hat_gradH_hat_T    = RB.T @ Xt @ gradH.T @ RB
                self.big_A_hat                 = RB.T @ A @ RB
                self.big_ic                    = RB.T @ x_0
                self.big_x0b4A                 = np.zeros_like(self.big_ic)
                if self.centered:
                    # compute the part needed for centering
                    self.big_x0b4A += RB.T @ A @ x_0
                    self.big_ic     = np.zeros_like(self.big_x0b4A)
            # now set the reduced operators
            gradH_hat_gradH_hat_T = self.big_gradH_hat_gradH_hat_T[:r,:r]
            Xt_hat_gradH_hat_T    = self.big_Xt_hat_gradH_hat_T[:r,:r]
            self.A_hat            = self.big_A_hat[:r,:r]
            self.ic               = self.big_ic[:r]
            self.x0b4A            = self.big_x0b4A[:r]
        else:
            # list bases have to recompute everything
            RB                    = self.basis_from_list(self.reduced_basis, r)
            gradH_hat_gradH_hat_T = RB.T @ gradH @ gradH.T @ RB
            Xt_hat_gradH_hat_T    = RB.T @ Xt @ gradH.T @ RB
            self.A_hat            = RB.T @ A @ RB
            self.ic               = RB.T @ x_0
            self.x0b4J            = np.zeros(r)
            if self.centered:
                self.x0b4J += RB.T @ A @ x_0
                self.ic     = np.zeros(r)

        P          = csc_matrix(np.kron(np.eye(r), gradH_hat_gradH_hat_T) 
                                + np.kron(gradH_hat_gradH_hat_T, np.eye(r)))
        rhs        = Xt_hat_gradH_hat_T - Xt_hat_gradH_hat_T.T    
        reg        = 2 * eps * identity(r**2)
        L_hat      = spsolve(P + reg, vec(rhs)).reshape((r,r), order='F')
        self.J_hat = 0.5 * (L_hat - L_hat.T)

    def infer_SKW_Hamiltonian(self, two_r, X, Xt, J):
        try: self.reduced_basis
        except: 
            print('need to set reduced basis first!')
            exit
        if self.centered:
            print('Sharma/Kramer/Wang H-OpInf only \
                   works with uncentered bases!')
            exit

        r  = two_r // 2
        x_0 = self.snapshots[:,0]

        # list bases have to recompute everything
        RB               = self.basis_from_list(self.reduced_basis, two_r)
        X_hat            = RB.T @ X
        Xt_hat           = RB.T @ Xt
        X1_hat_X1_hat_T  = X_hat[:r] @ X_hat[:r].T
        X2_hat_X2_hat_T  = X_hat[r:] @ X_hat[r:].T
        Xt1_hat_X2_hat_T = Xt_hat[:r] @ X_hat[r:].T
        Xt2_hat_X1_hat_T = Xt_hat[r:] @ X_hat[:r].T
        rhs1             =  Xt1_hat_X2_hat_T + Xt1_hat_X2_hat_T.T
        rhs2             = -Xt2_hat_X1_hat_T - Xt2_hat_X1_hat_T.T

        # Solving the two requisite sub-problems
        P21 = csc_matrix(np.kron(np.eye(r), X2_hat_X2_hat_T) 
                         + np.kron(X2_hat_X2_hat_T, np.eye(r)))
        P22 = csc_matrix(np.kron(np.eye(r), X1_hat_X1_hat_T) 
                         + np.kron(X1_hat_X1_hat_T, np.eye(r)))
        A11 = spsolve(P21, vec(rhs1)).reshape((r,r), order='F')
        A22 = spsolve(P22, vec(rhs2)).reshape((r,r), order='F')
        
        # Returning the block diagonal OpInf'd matrix
        Zb         = np.zeros((r,r))
        self.A_hat = csc_matrix(np.block([[A22, Zb], [Zb, A11]]))

        # Get J_hat and IC
        self.J_hat = RB.T @ J @ RB
        self.ic    = RB.T @ x_0


class Linear_Lagrangian_ROM(Linear_Variational_ROM):
    def __init__(self, snapshots):
        super(Linear_Lagrangian_ROM, self).__init__(snapshots)

    def assemble_Lagrangian_ROM(self, r, M, K, q_dot, q_ddot):
        try: self.reduced_basis
        except: 
            print('need to set reduced basis first!')
            exit

        q_0      = self.snapshots[:,0].flatten()
        q_dot_0  = q_dot[:,0].flatten()
        q_ddot_0 = q_ddot[:,0].flatten()

        # if not isinstance(self.reduced_basis, list):
        # check if operators have been precomputed
        try: self.big_M_hat, self.big_K_hat, self.big_ics
        except:
            # if not, compute them and store
            RB               = self.reduced_basis
            self.big_M_hat   = RB.T @ M @ RB
            self.big_K_hat   = RB.T @ K @ RB
            big_q_hat_dot_0  = RB.T @ q_dot_0
            big_q_hat_ddot_0 = RB.T @ q_ddot_0
            # big_q_hat_dot_0  = np.zeros(RB.shape[1])
            # big_q_hat_ddot_0  = np.zeros(RB.shape[1])
            big_q_hat_0      = np.zeros(RB.shape[1])
            self.big_x0b4K   = np.zeros(RB.shape[1])
            if self.centered:
                # compute the part needed for centering
                self.big_x0b4K   += RB.T @ K @ q_0
            else:
                big_q_hat_0  += RB.T @ q_0
                # big_q_hat_dot_0  += RB.T @ q_dot_0
                # big_q_hat_ddot_0 += RB.T @ q_ddot_0
            self.big_ics     = [big_q_hat_0, big_q_hat_dot_0, 
                                big_q_hat_ddot_0]
        # now set the reduced operators
        self.M_hat = self.big_M_hat[:r,:r]
        self.K_hat = self.big_K_hat[:r,:r]
        self.ics   = [self.big_ics[i][:r] for i in range(3)]
        self.x0b4K = self.big_x0b4K[:r]
        # else:
        #     # list bases have to recompute everything
        #     RB = self.basis_from_list(self.reduced_basis, r)
        #     self.M_hat = RB.T @ M @ RB
        #     self.K_hat = RB.T @ K @ RB
        #     self.ics   = [RB.T @ q_0, RB.T @ q_dot_0, RB.T @ q_ddot_0]
        #     self.x0b4K = np.zeros(r)
        #     if self.centered:
        #         self.x0b4K += RB.T @ K @ q_0
        #         self.ic     = np.zeros(r)
        #     if issparse(self.M_hat):
        #         self.M_hat = self.M_hat.todense()
        #         self.K_hat = self.K_hat.todense()

    def integrate_Lagrangian_ROM(self, times, gamma=1./2, beta=1./4):
        try: self.M_hat, self.K_hat
        except:
            print('Lagrangian ROM operators are not set!')
            
        r      = len(self.ics[0])
        dt     = times[1] - times[0]
        q      = np.zeros((r, times.shape[0]))
        q_dot  = np.zeros((r, times.shape[0]))
        q_ddot = np.zeros((r, times.shape[0]))
        q[:,0], q_dot[:,0], q_ddot[:,0] = self.ics

        LHS     = self.M_hat + beta * dt**2 * self.K_hat
        lu, piv = lu_factor(LHS)
        
        for i,t in enumerate(times[:-1]):
            rhs2 = self.K_hat @ (q[:,i] + dt * q_dot[:,i] 
                                 + (0.5-beta) * dt**2 * q_ddot[:,i])
            rhs  = -(self.x0b4K + rhs2)  ### is this correct?
            q_ddot[:,i+1] = lu_solve((lu, piv), rhs)
            q_dot[:,i+1]  = q_dot[:,i] + dt * ((1-gamma) * q_ddot[:,i] 
                                               + gamma * q_ddot[:,i+1])
            q[:,i+1]      = q[:,i] + dt * q_dot[:,i] \
                                   + dt**2 * ((0.5-beta) * q_ddot[:,i]
                                              + beta * q_ddot[:,i+1])
            
            self.q_hat, self.q_hat_dot, self.q_hat_ddot = q, q_dot, q_ddot
            
    def assemble_Hamiltonian_from_Lagrangian_ROM(self, r, M, K, q_dot):
        try: self.reduced_basis
        except: 
            print('need to set reduced basis first!')
            exit

        q_0     = self.snapshots[:,0]
        q_dot_0 = q_dot[:,0]

        if not isinstance(self.reduced_basis, list):
            # check if operators have been precomputed
            try: (self.big_M_hat, self.big_K_hat, self.big_q_0, 
                  self.big_qdot_0, self.big_q0b4K)
            except:
                # if not, compute them and store
                RB             = self.reduced_basis
                self.big_M_hat = RB.T @ M @ RB
                self.big_K_hat = RB.T @ K @ RB
                self.big_q_0   = RB.T @ q_0
                # self.big_p_0   = self.big_M_hat @ RB.T @ q_dot_0
                self.big_qdot_0 = RB.T @ q_dot_0
                self.big_q0b4K = np.zeros_like(self.big_q_0)
                if self.centered:
                    # compute the part needed for centering
                    self.big_q0b4K += RB.T @ K @ q_0
                    self.big_q_0    = np.zeros_like(self.big_q0b4K)
            # now set the reduced operators
            M_hat       = self.big_M_hat[:r,:r]
            M_hat_inv   = np.linalg.inv(M_hat)
            K_hat       = self.big_K_hat[:r,:r]
            ZZ          = np.zeros((r,r))
            self.JA_hat = csc_matrix(np.block([[ZZ, M_hat_inv],[-K_hat, ZZ]]))
            self.ic     = np.concatenate([self.big_q_0[:r], M_hat @ self.big_qdot_0[:r]])
            self.x0b4K  = np.concatenate([np.zeros(r), self.big_q0b4K[:r]])
        # else:
        #     # list bases have to recompute everything
        #     RB          = self.basis_from_list(self.reduced_basis, r)
        #     M_hat       = RB.T @ M @ RB
        #     K_hat       = RB.T @ K @ RB
        #     if issparse(M_hat):
        #         M_hat = M_hat.todense()
        #         K_hat = K_hat.todense()
        #     M_hat_inv   = np.linalg.inv(M_hat)
        #     ZZ          = np.zeros((r,r))
        #     self.JA_hat = csc_matrix(np.block([[ZZ, M_hat_inv],[-K_hat, ZZ]]))
        #     self.ic     = np.concatenate([RB.T @ q_0, M_hat @ RB.T @ q_dot_0])
        #     self.x0b4K  = np.zeros(2*r)
        #     if self.centered:
        #         self.x0b4K[r:] += RB.T @ K @ q_0
        #         self.ic[:r]     = np.zeros(r)

    def integrate_Hamiltonian_from_Lagrangian_ROM(self, times):
        try: two_r = self.JA_hat.shape[0]
        except:
            print('Hamiltonian ROM operators are not set!')    

        dt         = times[1] - times[0]
        x_hat      = np.zeros((two_r, times.shape[0]))
        x_hat[:,0] = self.ic
        LHS        = identity(two_r, format='csc') - dt/2 * self.JA_hat
        solve      = factorized(LHS)

        for i,t in enumerate(times[:-1]):
            rhs          = dt * (self.JA_hat @ x_hat[:,i] - self.x0b4K)
            delXi        = solve(rhs)
            x_hat[:,i+1] = x_hat[:,i] + delXi  
    
        r = two_r // 2
        self.q_hat, self.p_hat = x_hat[:r], x_hat[r:]


# Vectorize column-wise
def vec(A):
    m, n = A.shape[0], A.shape[1]
    return A.reshape(m*n, order='F')

# Compute Hamiltonian
def compute_Hamiltonian(data, A):
    Adata = A @ data
    result = np.einsum('ia,ia->a', Adata, data)
    return 0.5 * result


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

    LHat  = U.T @ L @ U
    # LHat  = U.T @ L @ U
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
                # LHat = U.T @ L @ U
                LHat = U.T @ L @ U
            else:
                # If OpList[0] has been replaced with an inferred Lhat
                LHat = L    
            AHat  = U.T @ A @ U
            x0part = np.zeros(n)
            if MC:
                x0part += U.T @ A @ ic.flatten()
        else:
            LAHat  = U.T @ L @ A @ U
            x0part = np.zeros(n)
            if MC:
                x0part += U.T @ L @ A @ ic.flatten()
    else:
        U = UU[:,:n]
        if Hamiltonian:
            LHat   = OpList[0][:n,:n]
            # LHat   = OpList[0][:n,:n]
            AHat   = OpList[1][:n,:n]
            x0part  = OpList[2][:n]
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
    if Hamiltonian:
        LHS = LHat + dt/2 * AHat
    else:
        LHS = np.eye(n) - dt/2 * LAHat
    lu, piv = lu_factor(LHS)

    # Integrate ROMs over test interval
    for i,t in enumerate(tTest[:-1]):
        if Hamiltonian:
            rhs     = -dt * (x0part + AHat @ xHat[:,i])
        else:
            rhs     = dt * (x0part + LAHat @ xHat[:,i])
        delxHati    = lu_solve((lu, piv), rhs)
        xHat[:,i+1] = xHat[:,i] + delxHati

    # Reconstruct FO solutions
    if MC:
        xRec = ic + U @ xHat
    else:
        xRec = U @ xHat

    return xRec


# Function which integrates C-H-OpInf ROM
def integrate_HOpInf_ROM(tTest, AhatOp, ic, UU, L):
    n  = AhatOp.shape[0]
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

    Lred = U.T @ L @ U
    LHSop = Lred + dt/2 * AhatOp
    rhsOp = -dt * AhatOp


    # Integrate ROMs over test interval
    lu, piv = lu_factor(LHSop)

    for i,t in enumerate(tTest[:-1]):
        rhs           = rhsOp @ xHatOp[:,i]
        delXhatOpi    = lu_solve((lu, piv), rhs)
        xHatOp[:,i+1] = xHatOp[:,i] + delXhatOpi

    # Reconstruct FO solutions
    XrecOp = U @ xHatOp

    return XrecOp


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
            # Lred  = U.T @ L @ U
            temp = U.T @ L @ U
            if issparse(temp):
                Lred  = -np.linalg.inv(temp.todense())
            else:
                Lred  = -np.linalg.inv(temp)
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
                  eps_x=0.01, eps_y=0.1, yLims=None, legend_loc=0,
                  cmap = "Dark2", colorIdx = range(10), save=True):
    n_t = arrs[0].shape[-1]
    
    # Define the update function that will be called at each iteration
    def update(lines, data, frame):
        for i,line in enumerate(lines):
            line.set_ydata(data[i][:,frame])
        return lines,

    fig, ax = plt.subplots()

    cmap = plt.get_cmap(cmap)
    colors = [cmap.colors[i] for i in colorIdx]

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
        lines[i], = ax.plot(xCoords, arrs[i][:,0], color=colors[i],
                            linestyle=styles[i], label=labels[i])
    plt.legend(loc=legend_loc)

    plotFunc = partial(update, lines, arrs)
    anim = FuncAnimation(fig, plotFunc, frames=n_t, interval=1)
    anim.save('myarray.gif', writer='imagemagick', fps=60) if save else None
    plt.show()