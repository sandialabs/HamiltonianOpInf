# Imports for exodus
import sys
sys.path.append('/Users/adgrube/Desktop/Code/seacas/lib')  ### replace with your Seacas installation
import exodus

# Standard imports
import numpy as np
from scipy.sparse import eye, csc_matrix
from scipy.sparse.linalg import spsolve, inv
from scipy.io import mmread
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})
plt.rcParams['figure.constrained_layout.use'] = True

# Import for utils
sys.path.append("..")

# import OpInf_utils as ou
import ROM_utils_check as ru

# # To suppress annoying umfpack "almost singular" warnings
# import warnings
# warnings.filterwarnings('ignore', '.*singular matrix!.*')

# Creating exodus file
fromFileName = 'beam_velIC_100.e'
toFileName   = 'beam_vecIC_100_written.e'

massName  = 'mass.mm'
stiffName = 'stiff.mm'

ROMvariables = ['exactSol_x', 'exactSol_y', 'exactSol_z',
                '1-disp_G-IntRom_x', '1-disp_G-IntRom_y', '1-disp_G-IntRom_z',
                '1-disp_G-OpRom_x', '1-disp_G-OpRom_y', '1-disp_G-OpRom_z',
                '1-disp_H-IntRom_x', '1-disp_H-IntRom_y', '1-disp_H-IntRom_z',
                '1-disp_C-H-OpRom_x', '1-disp_C-H-OpRom_y', '1-disp_C-H-OpRom_z',
                '2-disp_NC-H-OpRom_x', '2-disp_NC-H-OpRom_y', '2-disp_NC-H-OpRom_z',
                '2-disp_C-H-OpRom_x', '2-disp_C-H-OpRom_y', '2-disp_C-H-OpRom_z',
                '2-disp_G-IntRom_x', '2-disp_G-IntRom_y', '2-disp_G-IntRom_z',
                '2-disp_H-IntRom_x', '2-disp_H-IntRom_y', '2-disp_H-IntRom_z',
                '3-disp_G-OpRom_x', '3-disp_G-OpRom_y', '3-disp_G-OpRom_z',
                '3-disp_NC-H-OpRom_x', '3-disp_NC-H-OpRom_y', '3-disp_NC-H-OpRom_z',
                '3-disp_C-H-OpRom_x', '3-disp_C-H-OpRom_y', '3-disp_C-H-OpRom_z',
                '3-disp_G-IntRom_x', '3-disp_G-IntRom_y', '3-disp_G-IntRom_z',
                '3-disp_H-IntRom_x', '3-disp_H-IntRom_y', '3-disp_H-IntRom_z']

# # Use this the first time the file is created
# exo_copy = exodus.copyTransfer(fromFileName, toFileName,
#                                array_type='numpy',
#                                additionalNodalVariables=ROMvariables)

#Use this after file is created
exo_copy = exodus.exodus(f'{toFileName}', array_type='numpy')

# Function which grabs everything from the exodus file
# The snapshots are kind of weird -- energy drops after first iteration

def assemble_FOM(exo_file):
    # Import mass and stiffness matrices
    N     = exo_file.num_nodes()
    Nt    = exo_file.num_times()
    mass  = mmread(massName)
    stiff = mmread(stiffName)

    # # stiff = 0.5 * (stiff+stiff.T)
    # # mass  = 0.5 * (mass+mass.T)
    # M = mass.todense()
    # K = stiff.todense()


    # Build L matrix
    zz    = np.zeros((3*N, 3*N))
    ii    = np.eye(3*N)
    J     = np.block([[zz, ii], [-ii, zz]])

    # for grad H
    massInv = spsolve(mass.tocsc(), eye(3*N).tocsc())
    A       = np.block([[stiff.todense(), zz], [zz, massInv.todense()]])

    # Solution arrays ordered by node_id
    sol_x   = np.zeros((N, Nt))
    sol_y   = np.zeros((N, Nt))
    sol_z   = np.zeros((N, Nt))

    # for position
    for t in range(Nt):
        sol_x[:,t] = exo_file.get_node_variable_values('disp_x', t+1)
        sol_y[:,t] = exo_file.get_node_variable_values('disp_y', t+1)
        sol_z[:,t] = exo_file.get_node_variable_values('disp_z', t+1)

    # Full position array (interleaved like M, K matrices)
    q_arr       = np.zeros((3*N, Nt))
    q_arr[0::3] = sol_x
    q_arr[1::3] = sol_y
    q_arr[2::3] = sol_z

    # for derivative of position
    for t in range(Nt):
        sol_x[:,t] = exo_file.get_node_variable_values('solution_dot_x', t+1)
        sol_y[:,t] = exo_file.get_node_variable_values('solution_dot_y', t+1)
        sol_z[:,t] = exo_file.get_node_variable_values('solution_dot_z', t+1)

    # Full qDot array
    qDot_arr       = np.zeros((3*N, Nt))
    qDot_arr[0::3] = sol_x
    qDot_arr[1::3] = sol_y
    qDot_arr[2::3] = sol_z

    # Full momentum array (interleaved)
    p_arr = mass @ qDot_arr

    # for derivative of momentum
    for t in range(Nt):
        sol_x[:,t] = exo_file.get_node_variable_values('solution_dotdot_x', t+1)
        sol_y[:,t] = exo_file.get_node_variable_values('solution_dotdot_y', t+1)
        sol_z[:,t] = exo_file.get_node_variable_values('solution_dotdot_z', t+1)

    # Full pDot array
    pDot_arr       = np.zeros((3*N, Nt))
    pDot_arr[0::3] = sol_x
    pDot_arr[1::3] = sol_y
    pDot_arr[2::3] = sol_z
    pDot_arr       = mass @ pDot_arr

    # Build state, state derivative, and gradH
    x_arr     = np.concatenate((q_arr, p_arr), axis=0)
    xDot_arr  = np.concatenate((qDot_arr, pDot_arr), axis=0)
    gradH_arr = csc_matrix(A) @ x_arr

    return (x_arr, xDot_arr, gradH_arr), csc_matrix(J), csc_matrix(A)


if __name__ == "__main__":

    try: 
        mat       = np.load('plate_snaps.npy', allow_pickle=True)[()]
        xData     = mat['x']
        Xac       = mat['x_test']
        # Xac       = mat['x']
        xDotData  = mat['xDot']
        gradHData = mat['gradH']
        A       = mat['A_sparse']
        J       = mat['J_sparse']
    except:
        data, J, A = assemble_FOM(exo_copy)
        # Asp        = csc_matrix(A)
        # Jsp        = csc_matrix(J)
        tTrain     = np.linspace(0, 0.002, 201)
        xData, xDotData, gradHData = ru.integrate_Linear_HFOM(tTrain, 
                                                    data[0][:,0], J, A)
        tTest = np.linspace(0, 0.005, 501)
        Xac   = ru.integrate_Linear_HFOM(tTest, data[0][:,0], J, A)[0]
        dicto = {'x': xData, 'x_test': Xac, 'xDot': xDotData,
                'gradH': gradHData, 'A_sparse': A, 'J_sparse': J}
        np.save('plate_snaps.npy', dicto)

    # Rescaling the CL snapshots 
    N = len(xData) // 2
    normQ = np.linalg.norm(xData[:N])
    normP = np.linalg.norm(xData[N:])
    print(normQ, normP)
    xDataSC = np.zeros_like(xData)
    xDataSC[:N] = xData[:N] * normP/normQ
    xDataSC[N:] = xData[N:]
    normQ = np.linalg.norm(xDataSC[:N])
    normP = np.linalg.norm(xDataSC[N:])
    print(normQ, normP)

    POD   = ru.Linear_Hamiltonian_ROM(xData)
    PODmc = ru.Linear_Hamiltonian_ROM(xData)
    CL    = ru.Linear_Hamiltonian_ROM(xDataSC)
    CLmc  = ru.Linear_Hamiltonian_ROM(xDataSC)
    QP    = ru.Linear_Hamiltonian_ROM(xData)
    QPmc  = ru.Linear_Hamiltonian_ROM(xData)
    # QQ    = ru.Linear_Hamiltonian_ROM(xData)
    # QQmc  = ru.Linear_Hamiltonian_ROM(xData)
    SVD   = ru.Linear_Hamiltonian_ROM(xDataSC)
    SVDmc = ru.Linear_Hamiltonian_ROM(xDataSC)

    try:
        bases = np.load('plate_bases.npy', allow_pickle=True)[()]
        POD.reduced_basis, POD.basis_evals = bases['POD']
        POD.centered   = False
        PODmc.reduced_basis, PODmc.basis_evals = bases['PODmc']
        PODmc.centered = True
        CL.reduced_basis, CL.basis_evals = bases['CL']
        CL.centered    = False
        CLmc.reduced_basis, CLmc.basis_evals = bases['CLmc']
        CLmc.centered  = True
        QP.reduced_basis, QP.basis_evals = bases['QP']
        QP.centered    = False
        QPmc.reduced_basis, QPmc.basis_evals = bases['QPmc']
        QPmc.centered  = True
        SVD.reduced_basis, SVD.basis_evals = bases['SVD']
        SVD.centered   = False
        SVDmc.reduced_basis, SVDmc.basis_evals = bases['SVDmc']
        SVDmc.centered = True

    except:
        POD.set_reduced_basis('POD', rmax=125, randomized=True)
        PODmc.set_reduced_basis('POD', centered=True, rmax=125, randomized=True)
        CL.set_reduced_basis('cotangent_lift', rmax=125, randomized=True)
        CLmc.set_reduced_basis('cotangent_lift', centered=True, rmax=125, randomized=True)
        QP.set_reduced_basis('block_qp', rmax=125, randomized=True)
        QPmc.set_reduced_basis('block_qp', centered=True, rmax=125, randomized=True)
        # QQ.set_reduced_basis('block_qq', rmax=100, randomized=True)
        # QQmc.set_reduced_basis('block_qq', centered=True, rmax=100, randomized=True)
        SVD.set_reduced_basis('cotangent_lift', rmax=125, randomized=False)
        SVDmc.set_reduced_basis('cotangent_lift', centered=True, rmax=125, randomized=False)

        dicto = {'POD': [POD.reduced_basis, POD.basis_evals], 
                 'PODmc': [PODmc.reduced_basis, PODmc.basis_evals],
                 'CL': [CL.reduced_basis, CL.basis_evals], 
                 'CLmc': [CLmc.reduced_basis, CLmc.basis_evals],
                 'QP': [QP.reduced_basis, QP.basis_evals],
                 'QPmc': [QPmc.reduced_basis, QPmc.basis_evals],
                 'SVD': [SVD.reduced_basis, SVD.basis_evals],
                 'SVDmc': [SVDmc.reduced_basis, SVDmc.basis_evals]}
        np.save('plate_bases.npy', dicto)

    POD.compute_basis_energies()
    PODmc.compute_basis_energies()
    CL.compute_basis_energies()
    CLmc.compute_basis_energies()
    QP.compute_basis_energies()
    QPmc.compute_basis_energies()
    SVD.compute_basis_energies()
    SVDmc.compute_basis_energies()

    # SS, SSmc     = POD.basis_evals, PODmc.basis_evals
    # SS2, SS2mc   = CL.basis_evals, CLmc.basis_evals
    # SSq, SSp     = QP.basis_evals
    # SSqmc, SSpmc = QPmc.basis_evals

    # # DO LAGRANGIAN STUFF (since reviewer asked for it)
    # from scipy.sparse.linalg import inv
    # M    = csc_matrix(mmread(massName))
    # K    = csc_matrix(mmread(stiffName))
    # Qt   = xData[:N]
    # Qdt  = xDotData[:N]
    # Qddt = inv(M) @ xDotData[N:]
    # tTrain = np.linspace(0, 0.002, 201)
    # ics = (Qt[:,0], Qdt[:,0], Qddt[:,0])

    # Qt, Qdt, Qddt = ru.integrate_LFOM(tTrain, ics, M.todense(), np.zeros_like(M.todense()), K.todense(), 
    #                                   lambda x: np.zeros_like(x))
    # Pt = M @ Qdt

    # LAG = ru.Linear_Lagrangian_ROM(Qt)
    # LAG.set_reduced_basis('POD', centered=False, rmax=125, randomized=True)
    # LAG.compute_basis_energies()
    # LAGmc = ru.Linear_Lagrangian_ROM(Qt)
    # LAGmc.set_reduced_basis('POD', centered=True, rmax=125, randomized=True)
    # LAGmc.compute_basis_energies()

    nEigs = 80
    idx = [i+1 for i in range(nEigs)]

    name = "Dark2"
    cmap = plt.get_cmap(name)
    colors = ['lightsalmon','c','aquamarine','lightyellow']

    nList = [4*(i+1) for i in range(31)]
    errU     = np.zeros(len(nList))
    errUmc   = np.zeros(len(nList))
    errU2    = np.zeros(len(nList))
    errU2mc  = np.zeros(len(nList))
    errUqp   = np.zeros(len(nList))
    errUqpmc = np.zeros(len(nList))
    errUSVD   = np.zeros(len(nList))
    errUSVDmc = np.zeros(len(nList))
    errULag   = np.zeros(len(nList))
    errULagmc   = np.zeros(len(nList))
    for i,n in enumerate(nList):

        reconUmc    = PODmc.project(xData, n)
        reconU      = POD.project(xData, n)
        reconU2mc   = CLmc.project(xData, n)
        reconU2     = CL.project(xData, n)
        reconUqpmc  = QPmc.project(xData, n)
        reconUqp    = QP.project(xData, n)
        reconUSVDmc = SVDmc.project(xData, n)
        reconUSVD   = SVD.project(xData, n)
        
        # reconULagmc = np.zeros_like(reconUmc)
        # reconULagmc[:N] = LAGmc.project(Qt, n)
        # reconULagmc[N:] = M @ LAGmc.project(Qdt, n)
        # reconULag   = np.zeros_like(reconU)
        # reconULag[:N] = LAG.project(Qt, n)
        # reconULag[N:] = M @ LAG.project(Qdt, n)

        errUmc[i]   = ru.relError(xData, reconUmc)
        errU[i]     = ru.relError(xData, reconU)
        errU2mc[i]  = ru.relError(xData, reconU2mc)
        errU2[i]    = ru.relError(xData, reconU2)
        errUqpmc[i] = ru.relError(xData, reconUqpmc)
        errUqp[i]   = ru.relError(xData, reconUqp)
        errUSVDmc[i] = ru.relError(xData, reconUSVDmc)
        errUSVD[i]   = ru.relError(xData, reconUSVD)
        # errULagmc[i] = ru.relError(xData, reconULagmc)
        # errULag[i]   = ru.relError(xData, reconULag)

    from matplotlib.ticker import StrMethodFormatter

    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    ax[0].set_title('POD Snapshot Energy')
    ax[0].scatter(idx, POD.basis_energies[:nEigs], s=10., label='Ordinary POD (no MC)', color=cmap.colors[0])
    ax[0].scatter(idx, PODmc.basis_energies[:nEigs], s=20., label='Ordinary POD (MC)', marker='x', color=cmap.colors[0])
    ax[0].scatter(idx, CL.basis_energies[0][:nEigs], s=20, label='Cotangent Lift (no MC)', color=cmap.colors[1])
    ax[0].scatter(idx, CLmc.basis_energies[0][:nEigs], s=40., label='Cotangent Lift (MC)', marker='x', color=cmap.colors[1])
    ax[0].scatter(idx, SVD.basis_energies[1][:nEigs], s=10., label='Complex SVD (no MC)', color=cmap.colors[2])
    ax[0].scatter(idx, SVDmc.basis_energies[1][:nEigs], s=10., label='Complex SVD (MC)', marker='x', color=cmap.colors[2])
    ax[0].scatter(idx, QP.basis_energies[0][:nEigs], s=10., label='q Snapshots Only (no MC)', color=cmap.colors[3])
    ax[0].scatter(idx, QPmc.basis_energies[0][:nEigs], s=20., label='q Snapshots Only (MC)', marker='x', color=cmap.colors[3])
    # ax[0].legend(prop={'size': 8})

    ax[1].set_title('POD Projection Error')
    ax[1].semilogy(nList, errU, label='Ordinary POD (no MC)', marker='o', linestyle='-', markersize=5, color=cmap.colors[0])
    ax[1].semilogy(nList, errUmc, label='Ordinary POD (MC)', marker='x', linestyle='-.', markersize=6, color=cmap.colors[0])
    ax[1].semilogy(nList, errU2, label='Cotangent Lift (no MC)', marker='o', linestyle='-', markersize=8, color=cmap.colors[1])
    ax[1].semilogy(nList, errU2mc, label='Cotangent Lift (MC)', marker='x', linestyle='-.', markersize=9, color=cmap.colors[1])
    ax[1].semilogy(nList, errUSVD, label='Complex SVD (no MC)', marker='o', linestyle='-', markersize=5, color=cmap.colors[2])
    ax[1].semilogy(nList, errUSVDmc, label='Complex SVD (MC)', marker='x', linestyle='-.', markersize=6, color=cmap.colors[2])
    ax[1].semilogy(nList, errUqp, label='qp Block Basis (no MC)', marker='o', linestyle='-', markersize=5, color=cmap.colors[3])
    ax[1].semilogy(nList, errUqpmc, label='qp Block Basis (MC)', marker='x', linestyle='-.', markersize=6, color=cmap.colors[3])
    # ax[1].semilogy(nList, errULag, label='Lagrangian Basis (no MC)', marker='o', linestyle='-', markersize=5, color=cmap.colors[4])
    # ax[1].semilogy(nList, errULagmc, label='Lagrangian Basis (MC)', marker='x', linestyle='-.', markersize=6, color=cmap.colors[4])
    ax[1].set_title('POD Projection Error')
    # ax[1].set_ylabel('relative $L^2$ error')
    # ax[1].legend(prop={'size': 8})

    lines, labels = ax[1].get_legend_handles_labels()
    # ax[1].legend(prop={'size': 8})

    # # So far, nothing special except the managed prop_cycle. Now the trick:
    # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    ax[2].axis('off')
    ax[2].legend(lines, labels, loc='center left', ncol=1, prop={'size': 14})

    # ax[0].get_shared_y_axes().join(ax[0], ax[1])
    # ax[1].set_xticklabels([])
    for i in range(1):
        ax[i].minorticks_on()
        ax[i].yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax[i].set_ylabel('% POD energy')
    for i in range(2):
        ax[i].set_xlabel('basis size $n$')
    plt.tight_layout()
    plt.savefig('platePODenergy.pdf', transparent=True)
    plt.show(block=True)


    ### Exact solution
    Xac = xData
    NtTest = Xac.shape[1]
    Ttest  = 0.002
    # tTest  = np.linspace(0, Ttest, 501)
    tTest = np.linspace(0, Ttest, 201)
    # tTest = np.linspace(0, 0.02, 201)
    exactE = ru.compute_Hamiltonian(Xac, A)
    # N      = len(Xac[:,0])

    ### Parameters
    rom_list = [PODmc, CLmc, QPmc]
    nList    = [4*(i+1) for i in range(31)]

    fig, ax = plt.subplots(1, 3, figsize=(15,5), sharex=True, sharey=True)
    ax[0].set_ylabel('relative $L^2$ error')

    titleList = [f'Ordinary POD,  $T={Ttest}$', 
                f'Cotangent Lift,  $T={Ttest}$',
                f'Block $(q,p)$,  $T={Ttest}$']
    alpha=1

    try:
        solns = np.load('plate_solns.npy', allow_pickle=True)[()]
        XrecIntG    = solns['IntG']
        XrecOpGno   = solns['OpGno']
        XrecOpGre   = solns['OpGre']
        XrecIntHinc = solns['IntHinc']
        XrecIntHcon = solns['IntHcon']
        XrecOpHno   = solns['OpHno']
        XrecOpHre   = solns['OpHre']
        XrecIntG.shape[1] == len(nList)

    except:
        XrecIntG    = np.zeros((3, len(nList), 2*N, NtTest))
        XrecOpGno   = np.zeros((3, len(nList), 2*N, NtTest))
        XrecOpGre   = np.zeros((3, len(nList), 2*N, NtTest))
        XrecIntHinc = np.zeros((3, len(nList), 2*N, NtTest))
        XrecIntHcon = np.zeros((3, len(nList), 2*N, NtTest))
        XrecOpHno   = np.zeros((3, len(nList), 2*N, NtTest))
        XrecOpHre   = np.zeros((3, len(nList), 2*N, NtTest))

        data, J, A = assemble_FOM(exo_copy)

        for i in range(len(rom_list)):

            rom = rom_list[i]
            print(f'solutions {titleList[i]}')

            # This is a hack.  Should fix eventually....
            rom.A = A
            rom.J = J

            for j,n in enumerate(nList):
                print(f'n = {n}')

                rom.assemble_naive_ROM(n, J, A)
                try:
                    rom.integrate_naive_ROM(tTest) 
                    XrecIntG[i,j] = rom.decode(rom.x_hat)
                except: pass

                rom.infer_generic(n, xData, xDotData, eps=0.0e-8,
                                  reproject=False)
                try: 
                    rom.integrate_naive_ROM(tTest)
                    XrecOpGno[i,j] = rom.decode(rom.x_hat)
                except: pass
                rom.infer_generic(n, xData, xDotData, eps=0.0e-8, 
                                    reproject=True)
                try: 
                    rom.integrate_naive_ROM(tTest)
                    XrecOpGre[i,j] = rom.decode(rom.x_hat)
                except: pass

                rom.assemble_Hamiltonian_ROM(n, J, A)
                try:
                    rom.integrate_Hamiltonian_ROM(tTest, eps=0.0e-9, 
                                                inconsistent=True) 
                    XrecIntHinc[i,j] = rom.decode(rom.x_hat)
                except: pass
                try:
                    rom.integrate_Hamiltonian_ROM(tTest, eps=0.0e-9, 
                                                inconsistent=False) 
                    XrecIntHcon[i,j] = rom.decode(rom.x_hat)
                except: pass

                rom.infer_canonical_Hamiltonian(n, xData, xDotData, J, eps=0.0e-8, 
                                                old=False, reproject=False)
                try:
                    rom.integrate_Hamiltonian_ROM(tTest, eps=0.0e-9,
                                                inconsistent=False)
                    XrecOpHno[i,j] = rom.decode(rom.x_hat)
                except: pass
                rom.infer_canonical_Hamiltonian(n, xData, xDotData, J, eps=0.0e-8, 
                                                old=False, reproject=True)
                try:
                    rom.integrate_Hamiltonian_ROM(tTest, eps=0.0e-9,
                                                inconsistent=False)
                    XrecOpHre[i,j] = rom.decode(rom.x_hat)
                except: pass

        dicto = {'IntG': XrecIntG, 'OpGno': XrecOpGno, 
                 'OpGre': XrecOpGre, 'IntHinc': XrecIntHinc,
                 'IntHcon': XrecIntHcon, 'OpHno': XrecOpHno,
                 'OpHre': XrecOpHre}
        np.save('plate_solns.npy', dicto)

    try:
        errs = np.load('plate_errors.npy', allow_pickle=True)[()]
        eIntG, HamIntG       = errs['IntG']
        eOpGno, HamOpGno     = errs['OpGno']
        eOpGre, HamOpGre     = errs['OpGre']
        eIntHinc, HamIntHinc = errs['IntHinc']
        eIntHcon, HamIntHcon = errs['IntHcon']
        eOpHno, HamOpHno     = errs['OpHno']
        eOpHre, HamOpHre     = errs['OpHre']
        eIntG.shape[1] == len(nList)

    except:
        eIntG    = np.zeros((3, len(nList)))
        eOpGno   = np.zeros((3, len(nList)))
        eOpGre   = np.zeros((3, len(nList)))
        eIntHinc = np.zeros((3, len(nList)))
        eIntHcon = np.zeros((3, len(nList)))
        eOpHno   = np.zeros((3, len(nList)))
        eOpHre   = np.zeros((3, len(nList)))

        HamIntG    = np.zeros((3, len(nList), NtTest))
        HamOpGno   = np.zeros((3, len(nList), NtTest))
        HamOpGre   = np.zeros((3, len(nList), NtTest))
        HamIntHinc = np.zeros((3, len(nList), NtTest))
        HamIntHcon = np.zeros((3, len(nList), NtTest))
        HamOpHno   = np.zeros((3, len(nList), NtTest))
        HamOpHre   = np.zeros((3, len(nList), NtTest))

        for i in range(len(rom_list)):
            print(f'errors {titleList[i]}')

            for j,n in enumerate(nList):
                print(f'n = {n}')

                eIntG[i,j]    = ru.relError(Xac, XrecIntG[i,j])
                eOpGno[i,j]   = ru.relError(Xac, XrecOpGno[i,j])
                eOpGre[i,j]   = ru.relError(Xac, XrecOpGre[i,j])
                eIntHinc[i,j] = ru.relError(Xac, XrecIntHinc[i,j])
                eIntHcon[i,j] = ru.relError(Xac, XrecIntHcon[i,j])
                eOpHno[i,j]   = ru.relError(Xac, XrecOpHno[i,j])
                eOpHre[i,j]   = ru.relError(Xac, XrecOpHre[i,j])

                HamIntG[i,j]    = ru.compute_Hamiltonian(XrecIntG[i,j], A)
                HamIntG[i,j]    -= HamIntG[i,j][0]
                HamOpGno[i,j]   = ru.compute_Hamiltonian(XrecOpGno[i,j], A)
                HamOpGno[i,j]   -= HamOpGno[i,j][0]
                HamOpGre[i,j]   = ru.compute_Hamiltonian(XrecOpGre[i,j], A)
                HamOpGre[i,j]   -= HamOpGre[i,j][0]                
                HamIntHinc[i,j] = ru.compute_Hamiltonian(XrecIntHinc[i,j], A)
                HamIntHinc[i,j] -= HamIntHinc[i,j][0]
                HamIntHcon[i,j] = ru.compute_Hamiltonian(XrecIntHcon[i,j], A)
                HamIntHcon[i,j] -= HamIntHcon[i,j][0]
                HamOpHno[i,j]   = ru.compute_Hamiltonian(XrecOpHno[i,j], A)
                HamOpHno[i,j]   -= HamOpHno[i,j][0]
                HamOpHre[i,j]   = ru.compute_Hamiltonian(XrecOpHre[i,j], A)
                HamOpHre[i,j]   -= HamOpHre[i,j][0]

        dicto = {'IntG': [eIntG, HamIntG], 'OpGno': [eOpGno, HamOpGno], 
                 'OpGre': [eOpGre, HamOpGre], 'IntHinc': [eIntHinc, HamIntHinc],
                 'IntHcon': [eIntHcon, HamIntHcon], 'OpHno': [eOpHno, HamOpHno],
                 'OpHre': [eOpHre, HamOpHre]}
        np.save('plate_errors.npy', dicto)


    for i in range(len(rom_list)):

        # Print error magnitudes
        print(f'{i} the relative L2 errors for intrusive GROM are {eIntG[i]}')
        print(f'{i} the relative L2 errors for OpInf GROM (original) are {eOpGno[i]}')
        print(f'{i} the relative L2 errors for OpInf GROM (reprojected) are {eOpGre[i]}')    
        print(f'{i} the relative L2 errors for intrusive HROM (inconsistent) are {eIntHinc[i]}')
        print(f'{i} the relative L2 errors for intrusive HROM (consistent) are {eIntHcon[i]}')
        print(f'{i} the relative L2 errors for OpInf HROM (original) are {eOpHno[i]}')
        print(f'{i} the relative L2 errors for OpInf HROM (reprojected) are {eOpHre[i]}' + '\n')

        ax.flatten()[i].semilogy(nList, eIntG[i], 
                                    label='Intrusive G-ROM (MC)',
                                    marker='o', linestyle='-', linewidth=3.0, markersize=6, alpha=alpha,
                                    color = cmap.colors[0])
        ax.flatten()[i].semilogy(nList, eIntHinc[i],
                                    label='Intrusive H-ROM (MC, inconsistent)',
                                    marker='o', linestyle='-', linewidth=3.0, markersize=6, alpha=alpha,
                                    color=cmap.colors[2])
        ax.flatten()[i].semilogy(nList, eIntHcon[i],
                                    label='Intrusive H-ROM (MC, consistent)',
                                    marker='o', linestyle='-', linewidth=3.0, markersize=6, alpha=alpha,
                                    color=cmap.colors[4])
        ax.flatten()[i].semilogy(nList, eOpGno[i], 
                                    label='OpInf G-ROM (MC, original)',
                                    marker='p', linestyle='--', linewidth=2.0, markersize=6, alpha=alpha,
                                    color=cmap.colors[5])
        ax.flatten()[i].semilogy(nList, eOpHno[i], 
                                    label='OpInf H-ROM (MC, original)',
                                    marker='p', linestyle='--', linewidth=2.0, markersize=6, alpha=alpha,
                                    color=cmap.colors[6])
        ax.flatten()[i].semilogy(nList, eOpGre[i], 
                                    label='OpInf G-ROM (MC, reprojected)',
                                    marker='*', linestyle='-.', linewidth=1.0, markersize=6, alpha=alpha,
                                    color=cmap.colors[1])
        ax.flatten()[i].semilogy(nList, eOpHre[i], 
                                    label='OpInf H-ROM (MC, reprojected)',
                                    marker='*', linestyle='-.', linewidth=1.0, markersize=6, alpha=alpha,
                                    color=cmap.colors[3])

        # ax.flatten()[i].set_ylim([5*10**-3, 10**3])
        ax.flatten()[i].set_ylim([10**-5, 10**3])
        ax.flatten()[i].set_title(f'{titleList[i]}')
        # ax.flatten()[i].legend(prop={'size':8}, loc=3)
        ax.flatten()[i].set_xlabel('basis size $n$')

    plt.tight_layout()
    plt.savefig(f'PlatePlotT.pdf', transparent=True)
    plt.show()


    name = "Dark2"
    cmap = plt.get_cmap(name)
    name = "Set1"
    dmap = plt.get_cmap(name)

    plt.rcParams['axes.formatter.useoffset'] = True 

    fig, ax = plt.subplots(1, 3, figsize=(15,5), sharex=True)

    skip=10

    i,j = 0,15

    titleList   = [f'All Models, $n={4*(j+1)}$ modes',
                   'Conservative Models', 'Proposed Models']

    ax.flatten()[0].plot(tTest[::skip], exactE[::skip]-exactE[0], 
                                    label='FOM Solution',
                                    marker='o', linestyle='-', linewidth=3.0, markersize=6, alpha=alpha,
                                    color = dmap.colors[1])
    ax.flatten()[0].plot(tTest[::skip], HamIntG[i,j][::skip], 
                                    label='Intrusive G-ROM (MC)',
                                    marker='o', linestyle='-', linewidth=3.0, markersize=6, alpha=alpha,
                                    color = cmap.colors[0])
    ax.flatten()[0].plot(tTest[::skip], HamIntHinc[i,j][::skip],
                                label='Intrusive H-ROM (MC, inconsistent)',
                                marker='o', linestyle='-', linewidth=3.0, markersize=6, alpha=alpha,
                                color=cmap.colors[2])
    ax.flatten()[0].plot(tTest[::skip], HamIntHcon[i,j][::skip],
                                label='Intrusive H-ROM (MC, consistent)',
                                marker='o', linestyle='-', linewidth=3.0, markersize=6, alpha=alpha,
                                color=cmap.colors[4])
    ax.flatten()[0].plot(tTest[::skip], HamOpGno[i,j][::skip], 
                                label='OpInf G-ROM (MC, original)',
                                marker='p', linestyle='--', linewidth=2.0, markersize=6, alpha=alpha,
                                color=cmap.colors[5])
    ax.flatten()[0].plot(tTest[::skip], HamOpHno[i,j][::skip], 
                                label='OpInf H-ROM (MC, original)',
                                marker='p', linestyle='--', linewidth=2.0, markersize=6, alpha=alpha,
                                color=cmap.colors[6])
    ax.flatten()[0].plot(tTest[::skip], HamOpGre[i,j][::skip], 
                                label='OpInf G-ROM (MC, reprojected)',
                                marker='*', linestyle='-.', linewidth=1.0, markersize=6, alpha=alpha,
                                color=cmap.colors[1])
    ax.flatten()[0].plot(tTest[::skip], HamOpHre[i,j][::skip], 
                                label='OpInf H-ROM (MC, reprojected)',
                                marker='*', linestyle='-.', linewidth=1.0, markersize=6, alpha=alpha,
                                color=cmap.colors[3])

    ax.flatten()[1].plot(tTest[::skip], exactE[::skip]-exactE[0], 
                                    label='FOM Solution',
                                    marker='o', linestyle='-', linewidth=3.0, markersize=6, alpha=alpha,
                                    color = dmap.colors[1])
    # ax.flatten()[1].plot(tTest[::skip], HamIntG[i,j][::skip], 
    #                                 label='Intrusive G-ROM (MC)',
    #                                 marker='o', linestyle='-', linewidth=3.0, markersize=6, alpha=alpha,
    #                                 color = cmap.colors[0])
    ax.flatten()[1].plot(tTest[::skip], HamIntHinc[i,j][::skip],
                                label='Intrusive H-ROM (MC, inconsistent)',
                                marker='o', linestyle='-', linewidth=3.0, markersize=6, alpha=alpha,
                                color=cmap.colors[2])
    ax.flatten()[1].plot(tTest[::skip], HamIntHcon[i,j][::skip],
                                label='Intrusive H-ROM (MC, consistent)',
                                marker='o', linestyle='-', linewidth=3.0, markersize=6, alpha=alpha,
                                color=cmap.colors[4])
    # ax.flatten()[1].plot(tTest[::skip], HamOpGno[i,j][::skip], 
    #                             label='OpInf G-ROM (MC, original)',
    #                             marker='p', linestyle='--', linewidth=2.0, markersize=6, alpha=alpha,
    #                             color=cmap.colors[5])
    ax.flatten()[1].plot(tTest[::skip], HamOpHno[i,j][::skip], 
                                label='OpInf H-ROM (MC, original)',
                                marker='p', linestyle='--', linewidth=2.0, markersize=6, alpha=alpha,
                                color=cmap.colors[6])
    # ax.flatten()[1].plot(tTest[::skip], HamOpGre[i,j][::skip], 
    #                             label='OpInf G-ROM (MC, reprojected)',
    #                             marker='*', linestyle='-.', linewidth=1.0, markersize=6, alpha=alpha,
    #                             color=cmap.colors[1])
    ax.flatten()[1].plot(tTest[::skip], HamOpHre[i,j][::skip], 
                                label='OpInf H-ROM (MC, reprojected)',
                                marker='*', linestyle='-.', linewidth=1.0, markersize=6, alpha=alpha,
                                color=cmap.colors[3])



    ax.flatten()[2].plot(tTest[::skip], exactE[::skip]-exactE[0], 
                                    label='FOM Solution',
                                    marker='o', linestyle='-', linewidth=3.0, markersize=6, alpha=alpha,
                                    color = dmap.colors[1])
    # ax.flatten()[1].plot(tTest[::skip], HamIntG[i,j][::skip], 
    #                                 label='Intrusive G-ROM (MC)',
    #                                 marker='o', linestyle='-', linewidth=3.0, markersize=6, alpha=alpha,
    #                                 color = cmap.colors[0])
    # ax.flatten()[2].plot(tTest[::skip], HamIntHinc[i,j][::skip],
    #                             label='Intrusive H-ROM (MC, inconsistent)',
    #                             marker='o', linestyle='-', linewidth=3.0, markersize=6, alpha=alpha,
    #                             color=cmap.colors[2])
    ax.flatten()[2].plot(tTest[::skip], HamIntHcon[i,j][::skip],
                                label='Intrusive H-ROM (MC, consistent)',
                                marker='o', linestyle='-', linewidth=3.0, markersize=6, alpha=alpha,
                                color=cmap.colors[4])
    # ax.flatten()[1].plot(tTest[::skip], HamOpGno[i,j][::skip], 
    #                             label='OpInf G-ROM (MC, original)',
    #                             marker='p', linestyle='--', linewidth=2.0, markersize=6, alpha=alpha,
    #                             color=cmap.colors[5])
    # ax.flatten()[2].plot(tTest[::skip], HamOpHno[i,j][::skip], 
    #                             label='OpInf H-ROM (MC, original)',
    #                             marker='p', linestyle='--', linewidth=2.0, markersize=6, alpha=alpha,
    #                             color=cmap.colors[6])
    # ax.flatten()[1].plot(tTest[::skip], HamOpGre[i,j][::skip], 
    #                             label='OpInf G-ROM (MC, reprojected)',
    #                             marker='*', linestyle='-.', linewidth=1.0, markersize=6, alpha=alpha,
    #                             color=cmap.colors[1])
    ax.flatten()[2].plot(tTest[::skip], HamOpHre[i,j][::skip], 
                                label='OpInf H-ROM (MC, reprojected)',
                                marker='*', linestyle='-.', linewidth=1.0, markersize=6, alpha=alpha,
                                color=cmap.colors[3])

    ax[0].set_ylabel(r'$H(t) - H_0$')
    for i in range(3):
        ax[i].set_xlabel('time $t$')
        ax.flatten()[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ax.flatten()[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
        ax[i].set_title(titleList[i])
    # ax.flatten()[0].legend(prop={'size':8})

    lines, labels = ax.flatten()[0].get_legend_handles_labels()

    plt.tight_layout()
    plt.savefig(f'PlateEnergyT.pdf', transparent=True)
    plt.show()

    # GETTING LEGEND FIGURE
    fig, ax = plt.subplots(1, 1, figsize=(15,1))
    ax.axis('off')
    fig.legend(lines, labels, loc='center', ncol=3)
    plt.tight_layout()
    plt.savefig(f'PlateHamilLegend.pdf', transparent=True)
    plt.show()


    # fig, ax = plt.subplots(3, 1, figsize=(15,8), sharex=True)
    # dofs = [1764*i+1000 for i in range(3)]
    # labelList = ['x-displacement', 'y-displacement', 'z-displacement']
    # for i in range(3):
    #     ax[i].plot(tTest, Xac[dofs[i]], linewidth=2.5, color=cmap.colors[-1],
    #                label='FOM Solution')
    #     ax[i].plot(tTest, XrecIntHinc[0,-5][dofs[i]], linewidth=0.8, color=cmap.colors[1],
    #                label='Intrusive H-ROM (MC, inconsistent)')
    #     ax[i].plot(tTest, XrecIntHcon[0,-5][dofs[i]], linewidth=0.8, color=cmap.colors[0], 
    #                label='Intrusive H-ROM (MC, consistent)')
    #     # ax[i].plot(tTest, XrecOpHno[0,-5][dofs[i]], linewidth=0.8, color=cmap.colors[2], 
    #     #            label='OpInf H-ROM (MC, original)')
    #     # ax[i].plot(tTest, XrecOpHre[0,-5][dofs[i]], linewidth=0.8, color=cmap.colors[3], 
    #     #            label='OpInf H-ROM (MC, reprojected)')
    #     ax[i].axvline(0.002, color='b', linewidth=1.)
    #     ax[i].set_ylabel(labelList[i], fontsize=14)
    #     ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    # ax[2].set_xlabel('time $t$')
    # ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.35),
    #       fancybox=False, shadow=False, ncol=3)
    # fig.suptitle('Cantilever Plate: Displacement at a Single Point', y=1.1)
    # plt.tight_layout()
    # plt.savefig(f'PlateTrajectories.pdf', transparent=True)
    # plt.show()

    # tTest = tTrain
    # Xac = xData
    # exactE = ru.compute_Hamiltonian(Xac, A)

    # eIntLag  = np.zeros(len(nList))
    # eIntHam  = np.zeros(len(nList))
    # eIntHam2 = np.zeros(len(nList))

    # QrecIntLag = np.zeros((len(nList), N, len(tTest)))
    # PrecIntLag = np.zeros((len(nList), N, len(tTest)))
    # QrecIntHam = np.zeros((len(nList), N, len(tTest)))

    # HamIntLag  = np.zeros((len(nList), len(tTest)))

    # for j,n in enumerate(nList):

    #     LAG.assemble_Lagrangian_ROM(n, M, K, Qdt, Qddt)
    #     try:
    #         LAG.integrate_Lagrangian_ROM(tTest)
    #         QrecIntLag[j] = LAG.decode(LAG.q_hat)
    #         PrecIntLag[j] = M @ LAG.decode(LAG.q_hat_dot)
    #     except: pass
    #     eIntLag[j] = np.sqrt(ru.relError(QrecIntLag[j], Xac[:N])**2 + 
    #                          ru.relError(PrecIntLag[j], Xac[N:])**2)
        
    #     XrecIntLag = np.concatenate([QrecIntLag[j], PrecIntLag[j]], axis=0)
    #     HamIntLag[j]  = ru.compute_Hamiltonian(XrecIntLag, A) - exactE
    #     # lag_rom.assemble_Hamiltonian_from_Lagrangian_ROM(n, M, K, Qdt)
    #     # # try:
    #     # lag_rom.integrate_Hamiltonian_from_Lagrangian_ROM(tTest)
    #     # QrecIntHam[j] = lag_rom.decode(lag_rom.q_hat)
    #     # # except: pass
    #     # eIntHam[j]  = ru.relError(QrecIntHam[j], Xac[:N//2])
    #     # eIntHam2[j] = ru.relError(XrecIntHcon[0,j][:N//2], Xac[:N//2])
    #     # # eIntHam2[j] = eIntHcon[0,j]

    # # QrecIntLag = np.zeros((len(nList), int(N), len(tTest)))
    # # QrecIntHam = np.zeros((len(nList), int(N), len(tTest)))

    # # for j,n in enumerate(nList):

    # #     lag_rom.assemble_Lagrangian_ROM(n, M, K, Qdt, Qddt)
    # #     try:
    # #         lag_rom.integrate_Lagrangian_ROM(tTest)
    # #         QrecIntLag[j][:N//2] = lag_rom.decode(lag_rom.q_hat)
    # #         QrecIntLag[j][N//2:] = M @ lag_rom.decode(lag_rom.q_hat_dot)
    # #     except: pass
    # #     eIntLag[j] = ru.relError(QrecIntLag[j], Xac)
    # #     lag_rom.assemble_Hamiltonian_from_Lagrangian_ROM(n, M, K, Qdt)
    # #     # try:
    # #     lag_rom.integrate_Hamiltonian_from_Lagrangian_ROM(tTest)
    # #     QrecIntHam[j][:N//2] = lag_rom.decode(lag_rom.q_hat)
    # #     QrecIntHam[j][N//2:] = lag_rom.decode(lag_rom.p_hat)
    # #     # except: pass
    # #     eIntHam[j]  = ru.relError(QrecIntHam[j], Xac)
    # #     # eIntHam2[j] = ru.relError(XrecIntH[0,j][:N//2], Xac[:N//2])
    # #     eIntHam2[j] = eIntHcon[0,j]
    # eIntHam2 = eIntHcon[0]

    # fig, ax = plt.subplots(1, 1, figsize=(5,5))
    # ax.set_ylabel('relative $L^2$ error')

    # # Print error magnitudes
    # print(f'the relative L2 errors for Lagrangian GROM are {eIntLag}')
    # print(f'the relative L2 errors for Lagrangian HROM are {eIntHam}')
    # print(f'the relative L2 errors for intrusive HROM are {eIntHam2}')

    # ax.semilogy(nList, eIntLag, label='Lagrangian G-ROM', marker='o', linestyle='-', linewidth=0.5, markersize=6, color=cmap.colors[0])
    # # ax.semilogy(nList, eIntHam, label='Lagrangian H-ROM', marker='s', linestyle='-', linewidth=0.5, markersize=6, color=cmap.colors[1])
    # ax.semilogy(nList, eIntHam2, label='Hamiltonian H-ROM', marker='*', linestyle='-', linewidth=0.5, markersize=6, color=cmap.colors[2])
    # ax.set_ylim([10**-12, 5.])
    # # ax.set_title(f'{titleList[i]}')
    # ax.legend(prop={'size':8}, loc=3)
    # ax.set_xlabel('basis size $n$')

    # plt.tight_layout()
    # plt.savefig(f'PlatePlotT_hamlagham.pdf', transparent=True)
    # plt.show()

    # j=15

    # fig, ax = plt.subplots(1, 1, figsize=(5,5))

    # ax.plot(tTest[::skip], exactE[::skip]-exactE[0], 
    #                                 label='FOM Solution',
    #                                 marker='o', linestyle='-', linewidth=3.0, markersize=6, alpha=alpha,
    #                                 color = cmap.colors[-1])
    # ax.plot(tTest[::skip], HamIntLag[j][::skip], 
    #                                 label='Intrusive L-ROM (no MC)',
    #                                 marker='o', linestyle='-', linewidth=3.0, markersize=6, alpha=alpha,
    #                                 color = cmap.colors[0])
    # plt.show()