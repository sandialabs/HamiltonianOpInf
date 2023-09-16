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

# Import for utils
sys.path.append("..")

# import OpInf_utils as ou
import ROM_utils as ru

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

    # stiff = 0.5 * (stiff+stiff.T)
    # mass  = 0.5 * (mass+mass.T)
    M = mass.todense()
    K = stiff.todense()


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

    return (x_arr, xDot_arr, gradH_arr), J, A


# Compute Hamiltonian
def Hamil(data, A):
    result = np.zeros(data.shape[1])
    for i in range(len(result)):
        result[i] = data[:,i].T @ A @ data[:,i]
    return 0.5 * result



if __name__ == "__main__":

    try: 
        mat       = np.load('plate_snaps.npy', allow_pickle=True)[()]
        xData     = mat['x']
        Xac       = mat['x_test']
        # Xac       = mat['x']
        xDotData  = mat['xDot']
        gradHData = mat['gradH']
        Asp       = mat['A_sparse']
        Jsp       = mat['J_sparse']
    except:
        data, J, A = assemble_FOM(exo_copy)
        Asp        = csc_matrix(A)
        Jsp        = csc_matrix(J)
        tTrain     = np.linspace(0, 0.02, 201)
        xData, xDotData, gradHData = ru.integrate_Linear_HFOM(tTrain, 
                                                    data[0][:,0], Jsp, Asp)
        tTest = np.linspace(0, 0.1, 201)
        Xac   = ru.integrate_Linear_HFOM(tTrain, data[0][:,0], Jsp, Asp)[0]
        dicto = {'x': xData, 'x_test': Xac, 'xDot': xDotData,
                'gradH': gradHData, 'A_sparse': Asp, 'J_sparse': Jsp}
        np.save('plate_snaps.npy', dicto)


    POD   = ru.Linear_Hamiltonian_ROM(xData)
    PODmc = ru.Linear_Hamiltonian_ROM(xData)
    CL    = ru.Linear_Hamiltonian_ROM(xData)
    CLmc  = ru.Linear_Hamiltonian_ROM(xData)
    QP    = ru.Linear_Hamiltonian_ROM(xData)
    QPmc  = ru.Linear_Hamiltonian_ROM(xData)
    # QQ    = ru.Linear_Hamiltonian_ROM(xData)
    # QQmc  = ru.Linear_Hamiltonian_ROM(xData)

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
    except:
        POD.set_reduced_basis('POD', rmax=100, randomized=True)
        PODmc.set_reduced_basis('POD', centered=True, rmax=100, randomized=True)
        CL.set_reduced_basis('cotangent_lift', rmax=100, randomized=True)
        CLmc.set_reduced_basis('cotangent_lift', centered=True, rmax=100, randomized=True)
        QP.set_reduced_basis('block_qp', rmax=100, randomized=True)
        QPmc.set_reduced_basis('block_qp', centered=True, rmax=100, randomized=True)
        # QQ.set_reduced_basis('block_qq', rmax=100, randomized=True)
        # QQmc.set_reduced_basis('block_qq', centered=True, rmax=100, randomized=True)
        dicto = {'POD': [POD.reduced_basis, POD.basis_evals], 
                 'PODmc': [PODmc.reduced_basis, PODmc.basis_evals],
                 'CL': [CL.reduced_basis, CL.basis_evals], 
                 'CLmc': [CLmc.reduced_basis, CLmc.basis_evals],
                 'QP': [QP.reduced_basis, QP.basis_evals],
                 'QPmc': [QPmc.reduced_basis, QP.basis_evals]}
        np.save('plate_bases.npy', dicto)

    POD.compute_basis_energies()
    PODmc.compute_basis_energies()
    CL.compute_basis_energies()
    CLmc.compute_basis_energies()
    QP.compute_basis_energies()
    QPmc.compute_basis_energies()

    SS, SSmc     = POD.basis_evals, PODmc.basis_evals
    SS2, SS2mc   = CL.basis_evals, CLmc.basis_evals
    SSq, SSp     = QP.basis_evals
    SSqmc, SSpmc = QPmc.basis_evals

    nEigs = 25
    idx = [i+1 for i in range(nEigs)]

    name = "Set2"
    cmap = plt.get_cmap(name)
    colors = ['lightsalmon','c','aquamarine','lightyellow']

    nList = [4*(i+1) for i in range(25)]
    errU  = np.zeros(len(nList))
    errUmc = np.zeros(len(nList))
    errU2  = np.zeros(len(nList))
    errU2mc = np.zeros(len(nList))
    errUqp  = np.zeros(len(nList))
    errUqpmc = np.zeros(len(nList))
    for i,n in enumerate(nList):

        reconUmc   = PODmc.project(xData, n)
        reconU     = POD.project(xData, n)
        reconU2mc  = CLmc.project(xData, n)
        reconU2    = CL.project(xData, n)
        reconUqpmc = QPmc.project(xData, n)
        reconUqp   = QP.project(xData, n)

        errUmc[i]   = ru.relError(xData, reconUmc)
        errU[i]     = ru.relError(xData, reconU)
        errU2mc[i]  = ru.relError(xData, reconU2mc)
        errU2[i]    = ru.relError(xData, reconU2)
        errUqpmc[i] = ru.relError(xData, reconUqpmc)
        errUqp[i]   = ru.relError(xData, reconUqp)


    from matplotlib.ticker import StrMethodFormatter

    fig, ax = plt.subplots(1, 2, figsize=(9,3))
    ax[0].set_title('POD Snapshot Energy')
    ax[0].scatter(idx, POD.basis_energies[:nEigs], s=20., label='Ordinary POD (no MC)', color=cmap.colors[0])
    ax[0].scatter(idx, PODmc.basis_energies[:nEigs], s=40., label='Ordinary POD (with MC)', marker='x', color=cmap.colors[0])
    ax[0].scatter(idx, CL.basis_energies[0][:nEigs], s=10, label='Cotangent Lift (no MC)', color=cmap.colors[1])
    ax[0].scatter(idx, CLmc.basis_energies[0][:nEigs], s=20., label='Cotangent Lift (with MC)', marker='x', color=cmap.colors[1])
    ax[0].scatter(idx, QP.basis_energies[0][:nEigs], s=10., label='q Snapshots Only (no MC)', color=cmap.colors[2])
    ax[0].scatter(idx, QPmc.basis_energies[0][:nEigs], s=20., label='q Snapshots Only (with MC)', marker='x', color=cmap.colors[2])
    # ax[0].scatter(idx, QP.basis_energies[1][:nEigs], s=10., label='p Snapshots Only (no MC)', color=cmap.colors[4])
    # ax[0].scatter(idx, QPmc.basis_energies[1][:nEigs], s=10., label='p Snapshots Only (with MC)', marker='x', color=cmap.colors[4])
    ax[0].legend(prop={'size': 8})

    ax[1].set_title('POD Projection Error')
    ax[1].semilogy(nList, errU, label='Ordinary POD (no MC)', marker='o', linestyle='-', markersize=5, color=cmap.colors[0])
    ax[1].semilogy(nList, errUmc, label='Ordinary POD (MC)', marker='x', linestyle='-.', markersize=5, color=cmap.colors[0])
    ax[1].semilogy(nList, errU2, label='Cotangent Lift (no MC)', marker='o', linestyle='-', markersize=5, color=cmap.colors[1])
    ax[1].semilogy(nList, errU2mc, label='Cotangent Lift (MC)', marker='x', linestyle='-.', markersize=5, color=cmap.colors[1])
    ax[1].semilogy(nList, errUqp, label='qp Block Basis (no MC)', marker='o', linestyle='-', markersize=5, color=cmap.colors[2])
    ax[1].semilogy(nList, errUqpmc, label='qp Block Basis (MC)', marker='x', linestyle='-.', markersize=5, color=cmap.colors[2])
    ax[1].set_title('POD Projection Error')
    ax[1].set_ylabel('relative $L^2$ error')
    ax[1].legend(prop={'size': 8})

    # ax[0].get_shared_y_axes().join(ax[0], ax[1])
    # ax[1].set_xticklabels([])
    for i in range(1):
        ax[i].minorticks_on()
        ax[i].yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax[i].set_ylabel('% POD energy')
    for i in range(2):
        ax[i].set_xlabel('basis size $n$')
    plt.tight_layout()
    plt.savefig('platePODenergy', transparent=True)
    plt.show(block=False)


    # ### Exact solution
    # NtTest = Xac.shape[1]
    # tTest  = np.linspace(0, 1e-1, NtTest)
    # exactE = ru.compute_Hamiltonian(Xac, Asp)
    # N      = len(Xac[:,0])

    ### Exact solution
    data, J, A = assemble_FOM(exo_copy)
    Ttest     = 1e-1
    NtTest    = 201
    tTest     = np.linspace(0, Ttest, NtTest)
    Xac       = ru.integrate_Linear_HFOM(tTest, data[0][:,0], J, A)[0]
    exactE    = ru.compute_Hamiltonian(Xac, Asp)
    N         = len(data[0][:,0])

    ### Parameters
    rom_list = [PODmc, CLmc, QPmc]
    nList    = [4*(i+1) for i in range(10)]

    fig, ax = plt.subplots(1, 3, figsize=(15,5), sharex=True, sharey=True)
    ax[0].set_ylabel('relative $L^2$ error')

    titleList = ['Ordinary POD', 
                 'Cotangent Lift',
                 'Block $(q,p)$']
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
        XrecIntG    = np.zeros((3, len(nList), N, NtTest))
        XrecOpGno   = np.zeros((3, len(nList), N, NtTest))
        XrecOpGre   = np.zeros((3, len(nList), N, NtTest))
        XrecIntHinc = np.zeros((3, len(nList), N, NtTest))
        XrecIntHcon = np.zeros((3, len(nList), N, NtTest))
        XrecOpHno   = np.zeros((3, len(nList), N, NtTest))
        XrecOpHre   = np.zeros((3, len(nList), N, NtTest))

        # data, J, A = assemble_FOM(exo_copy)


        for i in range(len(rom_list)):

            rom = rom_list[i]
            print(titleList[i])

            # This is a hack.  Should fix eventually....
            rom.A = Asp

            for j,n in enumerate(nList):
                print(f'n = {n}')

                rom.assemble_naive_ROM(n, Jsp, Asp)
                try:
                    rom.integrate_naive_ROM(tTest) 
                    XrecIntG[i,j] = rom.decode(rom.x_hat)
                except: pass

                rom.assemble_Hamiltonian_ROM(n, Jsp, Asp)
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

                rom.infer_generic(n, xData, xDotData, eps=0.0e-5, reproject=False)
                try: 
                    rom.integrate_naive_ROM(tTest)
                    XrecOpGno[i,j] = rom.decode(rom.x_hat)
                except: pass
                rom.infer_generic(n, xData, xDotData, eps=0.0e-5, reproject=True)
                try: 
                    rom.integrate_naive_ROM(tTest)
                    XrecOpGre[i,j] = rom.decode(rom.x_hat)
                except: pass

                rom.infer_canonical_Hamiltonian(n, xData, xDotData, Jsp, eps=0.0e-5, 
                                                old=False, reproject=False)
                try:
                    rom.integrate_Hamiltonian_ROM(tTest, eps=0.0e-9,
                                                  inconsistent=False)
                    XrecOpHno[i,j] = rom.decode(rom.x_hat)
                except: pass
                rom.infer_canonical_Hamiltonian(n, xData, xDotData, Jsp, eps=0.0e-5, 
                                                old=False, reproject=True)
                try:
                    rom.integrate_Hamiltonian_ROM(tTest, eps=0.0e-9,
                                                  inconsistent=False)
                    XrecOpHre[i,j] = rom.decode(rom.x_hat)
                except: pass

                # rom.assemble_naive_ROM(n, Jsp, Asp)
                # try:
                #     rom.integrate_naive_ROM(tTest) 
                #     XrecIntG[i,j] = rom.decode(rom.x_hat)
                # except: pass

                # rom.infer_generic(n, xData, xDotData, eps=0.0e-5, reproject=False)
                # try: 
                #     rom.integrate_naive_ROM(tTest)
                #     XrecOpGno[i,j] = rom.decode(rom.x_hat)
                # except: pass
                # rom.infer_generic(n, xData, xDotData, eps=0.0e-5, reproject=True)
                # try: 
                #     rom.integrate_naive_ROM(tTest)
                #     XrecOpGre[i,j] = rom.decode(rom.x_hat)
                # except: pass

                # rom.assemble_Hamiltonian_ROM(n, Jsp, Asp)
                # try:
                #     rom.integrate_Hamiltonian_ROM(tTest, eps=0.0e-9, 
                #                                   inconsistent=True) 
                #     XrecIntHinc[i,j] = rom.decode(rom.x_hat)
                # except: pass
                # try:
                #     rom.integrate_Hamiltonian_ROM(tTest, eps=0.0e-9, 
                #                                   inconsistent=False) 
                #     XrecIntHcon[i,j] = rom.decode(rom.x_hat)
                # except: pass

                # rom.infer_canonical_Hamiltonian(n, xData, xDotData, Jsp, eps=0.0e-5, 
                #                                 old=False, reproject=False)
                # try:
                #     rom.integrate_Hamiltonian_ROM(tTest, eps=0.0e-9,
                #                                   inconsistent=False)
                #     XrecOpHno[i,j] = rom.decode(rom.x_hat)
                # except: pass
                # rom.infer_canonical_Hamiltonian(n, xData, xDotData, Jsp, eps=0.0e-5, 
                #                                 old=False, reproject=True)
                # try:
                #     rom.integrate_Hamiltonian_ROM(tTest, eps=0.0e-9,
                #                                   inconsistent=False)
                #     XrecOpHre[i,j] = rom.decode(rom.x_hat)
                # except: pass

        dicto = {'IntG': XrecIntG, 'OpGno': XrecOpGno, 
                 'OpGre': XrecOpGre, 'IntHinc': XrecIntHinc,
                 'IntHcon': XrecIntHcon, 'OpHno': XrecOpHno,
                 'OpHre': XrecOpHre}
        np.save('plate_solns.npy', dicto)

    eIntG    = np.zeros((3, len(nList)))
    eOpGno   = np.zeros((3, len(nList)))
    eOpGre   = np.zeros((3, len(nList)))
    eIntHinc = np.zeros((3, len(nList)))
    eIntHcon = np.zeros((3, len(nList)))
    eOpHno   = np.zeros((3, len(nList)))
    eOpHre   = np.zeros((3, len(nList)))

    HamIntG    = np.zeros((3, len(nList), N, NtTest))
    HamOpGno   = np.zeros((3, len(nList), N, NtTest))
    HamOpGre   = np.zeros((3, len(nList), N, NtTest))
    HamIntHinc = np.zeros((3, len(nList), N, NtTest))
    HamIntHcon = np.zeros((3, len(nList), N, NtTest))
    HamOpHno   = np.zeros((3, len(nList), N, NtTest))
    HamOpHre   = np.zeros((3, len(nList), N, NtTest))

    for i in range(len(rom_list)):
        for j,n in enumerate(nList):

            eIntG[i,j]    = ru.relError(Xac, XrecIntG[i,j])
            eOpGno[i,j]   = ru.relError(Xac, XrecOpGno[i,j])
            eOpGre[i,j]   = ru.relError(Xac, XrecOpGre[i,j])
            eIntHinc[i,j] = ru.relError(Xac, XrecIntHinc[i,j])
            eIntHcon[i,j] = ru.relError(Xac, XrecIntHcon[i,j])
            eOpHno[i,j]   = ru.relError(Xac, XrecOpHno[i,j])
            eOpHre[i,j]   = ru.relError(Xac, XrecOpHre[i,j])

            HamIntG[i,j]    = ru.compute_Hamiltonian(XrecIntG[i,j], Asp) - exactE
            HamOpGno[i,j]   = ru.compute_Hamiltonian(XrecOpGno[i,j], Asp) - exactE
            HamOpGre[i,j]   = ru.compute_Hamiltonian(XrecOpGre[i,j], Asp) - exactE
            HamIntHinc[i,j] = ru.compute_Hamiltonian(XrecIntHinc[i,j], Asp) - exactE
            HamIntHcon[i,j] = ru.compute_Hamiltonian(XrecIntHcon[i,j], Asp) - exactE
            HamOpHno[i,j]   = ru.compute_Hamiltonian(XrecOpHno[i,j], Asp) - exactE
            HamOpHre[i,j]   = ru.compute_Hamiltonian(XrecOpHre[i,j], Asp) - exactE

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
                                 marker='o', linestyle='-', linewidth=0.5, markersize=6, alpha=alpha)
        ax.flatten()[i].semilogy(nList, eOpGno[i], 
                                 label='OpInf G-ROM (MC, original)',
                                 marker='s', linestyle='--', linewidth=0.5, markersize=6, alpha=alpha)
        ax.flatten()[i].semilogy(nList, eOpGre[i], 
                                 label='OpInf G-ROM (MC, reprojected)',
                                 marker='s', linestyle='-', linewidth=0.5, markersize=6, alpha=alpha)
        ax.flatten()[i].semilogy(nList, eIntHinc[i],
                                 label='Intrusive H-ROM (MC, inconsistent)',
                                 marker='*', linestyle='--', linewidth=0.5, markersize=6, alpha=alpha)
        ax.flatten()[i].semilogy(nList, eIntHcon[i],
                                 label='Intrusive H-ROM (MC, consistent)',
                                 marker='*', linestyle='-', linewidth=0.5, markersize=6, alpha=alpha)
        ax.flatten()[i].semilogy(nList, eOpHno[i], 
                                 label='OpInf H-ROM (MC, original)',
                                 marker='s', linestyle='--', linewidth=0.5, markersize=6, alpha=alpha)
        ax.flatten()[i].semilogy(nList, eOpHre[i], 
                                 label='OpInf H-ROM (MC, reprojected)',
                                 marker='s', linestyle='-', linewidth=0.5, markersize=6, alpha=alpha)
        ax.flatten()[i].set_ylim([10**-12, 10.])
        ax.flatten()[i].set_title(f'{titleList[i]}')
        ax.flatten()[i].legend(prop={'size':8}, loc=2)
        ax.flatten()[i].set_xlabel('basis size $n$')

    plt.tight_layout()
    plt.savefig(f'PlatePlotT', transparent=True)
    plt.show()