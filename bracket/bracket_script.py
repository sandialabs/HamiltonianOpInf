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
fromFileName = "bracket_velIC_100.e"
toFileName   = "bracket_velIC_100_written.e"

massName  = 'mass.mm'
stiffName = 'stiff.mm'

ROMvariables = ['disp_trueSoln_x', 'disp_trueSoln_y', 'disp_trueSoln_z',
                'disp_IntRomG_x', 'disp_IntRomG_y', 'disp_IntRomG_z',
                'disp_OpRomG_x', 'disp_OpRomG_y', 'disp_OpRomG_z',
                'disp_IntRomH_x', 'disp_IntRomH_y', 'disp_IntRomH_z',
                'disp_OpRomH_x', 'disp_OpRomH_y', 'disp_OpRomH_z']

try:
    # Use this the first time the file is created
    exo_copy = exodus.copyTransfer(fromFileName, toFileName,
                                array_type='numpy',
                                additionalNodalVariables=ROMvariables)
except:
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
        mat       = np.load('bracket_snaps.npy', allow_pickle=True)[()]
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
        Xac   = ru.integrate_Linear_HFOM(tTest, data[0][:,0], Jsp, Asp)[0]
        dicto = {'x': xData, 'x_test': Xac, 'xDot': xDotData,
                'gradH': gradHData, 'A_sparse': Asp, 'J_sparse': Jsp}
        np.save('bracket_snaps.npy', dicto)


    POD   = ru.Linear_Hamiltonian_ROM(xData)
    PODmc = ru.Linear_Hamiltonian_ROM(xData)
    CL    = ru.Linear_Hamiltonian_ROM(xData)
    CLmc  = ru.Linear_Hamiltonian_ROM(xData)
    QP    = ru.Linear_Hamiltonian_ROM(xData)
    QPmc  = ru.Linear_Hamiltonian_ROM(xData)
    # QQ    = ru.Linear_Hamiltonian_ROM(xData)
    # QQmc  = ru.Linear_Hamiltonian_ROM(xData)

    try:
        bases = np.load('bracket_bases.npy', allow_pickle=True)[()]
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
        np.save('bracket_bases.npy', dicto)

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
    plt.savefig('bracketPODenergy', transparent=True)
    plt.show(block=False)


    ### Exact solution
    NtTest = Xac.shape[1]
    tTest  = np.linspace(0, 1e-1, NtTest)
    exactE = ru.compute_Hamiltonian(Xac, Asp)
    N      = len(Xac[:,0])

    ### Parameters
    rom_list = [PODmc, CLmc, QPmc]
    nList    = [4*(i+1) for i in range(25)]

    titleList = ['Ordinary POD', 
                 'Cotangent Lift',
                 'Block $(q,p)$']
    alpha=1

    try:
        solns = np.load('bracket_solns.npy', allow_pickle=True)[()]
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

        data, J, A = assemble_FOM(exo_copy)

        for i in range(len(rom_list)):

            rom = rom_list[i]
            print(f'solutions {titleList[i]}')

            # This is a hack.  Should fix eventually....
            rom.A = Asp

            for j,n in enumerate(nList):
                print(f'n = {n}')

                rom.assemble_naive_ROM(n, J, A)
                try:
                    rom.integrate_naive_ROM(tTest) 
                    XrecIntG[i,j] = rom.decode(rom.x_hat)
                except: pass

                rom.infer_generic(n, xData, xDotData, eps=0.0e-5,
                                  reproject=False)
                try: 
                    rom.integrate_naive_ROM(tTest)
                    XrecOpGno[i,j] = rom.decode(rom.x_hat)
                except: pass
                rom.infer_generic(n, xData, xDotData, eps=0.0e-5, 
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

                rom.infer_canonical_Hamiltonian(n, xData, xDotData, J, eps=0.0e-5, 
                                                old=False, reproject=False)
                try:
                    rom.integrate_Hamiltonian_ROM(tTest, eps=0.0e-9,
                                                inconsistent=False)
                    XrecOpHno[i,j] = rom.decode(rom.x_hat)
                except: pass
                rom.infer_canonical_Hamiltonian(n, xData, xDotData, J, eps=0.0e-5, 
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
        np.save('bracket_solns.npy', dicto)

    try:
        errs = np.load('bracket_errors.npy', allow_pickle=True)[()]
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

        HamIntG    = np.zeros((3, len(nList), N, NtTest))
        HamOpGno   = np.zeros((3, len(nList), N, NtTest))
        HamOpGre   = np.zeros((3, len(nList), N, NtTest))
        HamIntHinc = np.zeros((3, len(nList), N, NtTest))
        HamIntHcon = np.zeros((3, len(nList), N, NtTest))
        HamOpHno   = np.zeros((3, len(nList), N, NtTest))
        HamOpHre   = np.zeros((3, len(nList), N, NtTest))

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

                HamIntG[i,j]    = ru.compute_Hamiltonian(XrecIntG[i,j], Asp) - exactE
                HamOpGno[i,j]   = ru.compute_Hamiltonian(XrecOpGno[i,j], Asp) - exactE
                HamOpGre[i,j]   = ru.compute_Hamiltonian(XrecOpGre[i,j], Asp) - exactE
                HamIntHinc[i,j] = ru.compute_Hamiltonian(XrecIntHinc[i,j], Asp) - exactE
                HamIntHcon[i,j] = ru.compute_Hamiltonian(XrecIntHcon[i,j], Asp) - exactE
                HamOpHno[i,j]   = ru.compute_Hamiltonian(XrecOpHno[i,j], Asp) - exactE
                HamOpHre[i,j]   = ru.compute_Hamiltonian(XrecOpHre[i,j], Asp) - exactE

        dicto = {'IntG': [eIntG, HamIntG], 'OpGno': [eOpGno, HamOpGno], 
                 'OpGre': [eOpGre, HamOpGre], 'IntHinc': [eIntHinc, HamIntHinc],
                 'IntHcon': [eIntHcon, HamIntHcon], 'OpHno': [eOpHno, HamOpHno],
                 'OpHre': [eOpHre, HamOpHre]}
        np.save('bracket_errors.npy', dicto)


    fig, ax = plt.subplots(1, 3, figsize=(15,5), sharex=True, sharey=True)
    ax[0].set_ylabel('relative $L^2$ error')

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
                                    marker='o', linestyle='-', linewidth=0.5,
                                    markersize=6, color=cmap.colors[0])
        ax.flatten()[i].semilogy(nList, eOpGno[i], 
                                    label='OpInf G-ROM (MC, original)',
                                    marker='o', linestyle='--', linewidth=0.5,
                                    markersize=3, color=cmap.colors[1])
        ax.flatten()[i].semilogy(nList, eOpGre[i], 
                                    label='OpInf G-ROM (MC, reprojected)',
                                    marker='o', linestyle='-', linewidth=0.5,
                                    markersize=3, color=cmap.colors[2])
        ax.flatten()[i].semilogy(nList, eIntHinc[i],
                                    label='Intrusive H-ROM (MC, inconsistent)',
                                    marker='s', linestyle='--', linewidth=0.5,
                                    markersize=6, color=cmap.colors[3])
        ax.flatten()[i].semilogy(nList, eIntHcon[i],
                                    label='Intrusive H-ROM (MC, consistent)',
                                    marker='s', linestyle='-', linewidth=0.5,
                                    markersize=6, color=cmap.colors[4])
        ax.flatten()[i].semilogy(nList, eOpHno[i], 
                                    label='OpInf H-ROM (MC, original)',
                                    marker='s', linestyle='--', linewidth=0.5,
                                    markersize=3, color=cmap.colors[5])
        ax.flatten()[i].semilogy(nList, eOpHre[i], 
                                    label='OpInf H-ROM (MC, reprojected)',
                                    marker='s', linestyle='-', linewidth=0.5,
                                    markersize=3, color=cmap.colors[6])
        ax.flatten()[i].set_ylim([10**-12, 10.])
        ax.flatten()[i].set_title(f'{titleList[i]}')
        ax.flatten()[i].legend(prop={'size':8}, loc=3)
        ax.flatten()[i].set_xlabel('basis size $n$')

    plt.tight_layout()
    # plt.savefig(f'BracketPlotT', transparent=True)
    plt.show()


    ########## Extra Plot for Irina ###########
    n = 20
    cmap = plt.get_cmap("Dark2")

    fig, ax = plt.subplots(1, 1, figsize=(5,5), sharex=True, sharey=True)
    ax.set_ylabel('relative $L^2$ error')

    ax.semilogy(nList[:n], eIntG[0][:n], 
                                label='Intrusive G-ROM (MC)',
                                marker='o', linestyle='-', linewidth=1.5,
                                markersize=6, color=cmap.colors[0])
    # ax.semilogy(nList[:n], eOpGno[i][:n], 
    #                             label='OpInf G-ROM (MC, original)',
    #                             marker='o', linestyle='--', linewidth=0.5,
    #                             markersize=3, color=cmap.colors[1])
    ax.semilogy(nList[:n], eOpGre[0][:n], 
                                label='OpInf G-ROM (MC, reprojected)',
                                marker='o', linestyle='--', linewidth=0.5,
                                markersize=2, color=cmap.colors[1])
    # ax.semilogy(nList[:n], eIntHinc[0][:n],
    #                             label='Intrusive H-ROM (MC, inconsistent)',
    #                             marker='s', linestyle='--', linewidth=0.5,
    #                             markersize=6, color=cmap.colors[3])
    ax.semilogy(nList[:n], eIntHcon[0][:n],
                                label='Intrusive H-ROM (MC, consistent)',
                                marker='s', linestyle='-', linewidth=1.5,
                                markersize=6, color=cmap.colors[2])
    # ax.semilogy(nList[:n], eOpHno[0][:n], 
    #                             label='OpInf H-ROM (MC, original)',
    #                             marker='s', linestyle='--', linewidth=0.5,
    #                             markersize=3, color=cmap.colors[5])
    ax.semilogy(nList[:n], eOpHre[0][:n], 
                                label='OpInf H-ROM (MC, reprojected)',
                                marker='s', linestyle='--', linewidth=0.5,
                                markersize=2, color=cmap.colors[3])
    ax.set_ylim([10**-9, 10.])
    ax.set_title(f'{titleList[0]}')
    ax.legend(prop={'size':8}, loc=3)
    ax.set_xlabel('basis size $n$')

    plt.tight_layout()
    plt.savefig(f'For_Irina_bracket', transparent=True)
    plt.show()


    N, i, j = exo_copy.num_nodes(), 0, 16

    # disp_xTest  = Xac[::3][:N,:]
    # disp_yTest  = Xac[1::3][:N,:]
    # disp_zTest  = Xac[2::3][:N,:]

    # disp_xIntG = XrecIntG[i,j][::3][:N,:]
    # disp_yIntG = XrecIntG[i,j][1::3][:N,:]
    # disp_zIntG = XrecIntG[i,j][2::3][:N,:]

    # disp_xOpG  = XrecOpGre[i,j][::3][:N,:]
    # disp_yOpG  = XrecOpGre[i,j][1::3][:N,:]
    # disp_zOpG  = XrecOpGre[i,j][2::3][:N,:]

    # disp_xIntH = XrecIntHcon[i,j][::3][:N,:]
    # disp_yIntH = XrecIntHcon[i,j][1::3][:N,:]
    # disp_zIntH = XrecIntHcon[i,j][2::3][:N,:]

    # disp_xOpH  = XrecOpHre[i,j][::3][:N,:]
    # disp_yOpH  = XrecOpHre[i,j][1::3][:N,:]
    # disp_zOpH  = XrecOpHre[i,j][2::3][:N,:]

    # for i in range(exo_copy.num_times()):

    #     exo_copy.put_node_variable_values("disp_trueSoln_x", i+1, disp_xTest[:,i])
    #     exo_copy.put_node_variable_values("disp_trueSoln_y", i+1, disp_yTest[:,i])
    #     exo_copy.put_node_variable_values("disp_trueSoln_z", i+1, disp_zTest[:,i])

    #     exo_copy.put_node_variable_values("disp_IntRomG_x", i+1, disp_xIntG[:,i])
    #     exo_copy.put_node_variable_values("disp_IntRomG_y", i+1, disp_yIntG[:,i])
    #     exo_copy.put_node_variable_values("disp_IntRomG_z", i+1, disp_zIntG[:,i])

    #     exo_copy.put_node_variable_values("disp_OpRomG_x", i+1, disp_xOpG[:,i])
    #     exo_copy.put_node_variable_values("disp_OpRomG_y", i+1, disp_yOpG[:,i])
    #     exo_copy.put_node_variable_values("disp_OpRomG_z", i+1, disp_zOpG[:,i])

    #     exo_copy.put_node_variable_values("disp_IntRomH_x", i+1, disp_xIntH[:,i])
    #     exo_copy.put_node_variable_values("disp_IntRomH_y", i+1, disp_yIntH[:,i])
    #     exo_copy.put_node_variable_values("disp_IntRomH_z", i+1, disp_zIntH[:,i])

    #     exo_copy.put_node_variable_values("disp_OpRomH_x", i+1, disp_xOpH[:,i])
    #     exo_copy.put_node_variable_values("disp_OpRomH_y", i+1, disp_yOpH[:,i])
    #     exo_copy.put_node_variable_values("disp_OpRomH_z", i+1, disp_zOpH[:,i])

    print(eIntG[0][16], eIntHcon[0][16])