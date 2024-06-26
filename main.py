import numpy as np
from scipy import special
from geometries import cubic_domain
from operators import fonction_spaces, solve_, solve_WCAWE, integral_mesure, z_f_matrices_b1p_withDiff, basis_N_WCAWE, Zn_Fn_matrices, reduced_solution
from postprocess import relative_errZ,import_FOM_result
from dolfinx.fem import (form, Function, petsc)
import petsc4py
from petsc4py import PETSc
import matplotlib.pyplot as plt
import time



rho0 = 1.21
c0 = 343.8

side_box = 0.40
radius = 0.1
freqvec = np.arange(80, 2001, 20)
bgt_order = "B1P"
geometry1 = "cubic"
geometry2 = "large"
geometry = geometry1 +'_'+ geometry2

mesh_info, submesh_info  = cubic_domain(side_box, radius, lc=1.5e-2)
dx, ds, dx1 = integral_mesure(mesh_info, submesh_info) # The ds value will be important to obtain the pression over the piston, dx and dx1 are useless here.
if bgt_order == "B1P":
    deg = 1
else:
    print('implement other order')
P, Q = fonction_spaces(mesh_info, submesh_info, deg)


print('---------------------------------------------------------------------------')
print('\t\tB1p modelisation, classical frequency sweep')
print('---------------------------------------------------------------------------')

from_data = True
if from_data:
    ###
    print("Data taken from "+bgt_order+'_'+geometry+".txt")
    
    
    surfarea = petsc.assemble.assemble_scalar(form(1*ds(1)))
else:
    ### In this loop, a frequency sweep is performed based on the programs in Baffle_study folder
    geometry = geometry1 + " with side box = " + str(side_box)
    
    Pav1 = np.zeros(freqvec.size,dtype=np.complex_)
    
    print('The geometry called has a vibrating surface area such as : ')
    surfarea = petsc.assemble.assemble_scalar(form(1*ds(1)))
    print(surfarea)
    t1 = time.time()
    Psol1, Qsol1 = Function(P), Function(Q)
    offset = P.dofmap.index_map.size_local * P.dofmap.index_map_bs
    for ii in range(freqvec.size):
        freq = freqvec[ii]
        if freq%100==0:
            print('current freq : ')
            print(freq)
        list_Z, list_F = z_f_matrices_b1p_withDiff(mesh_info, submesh_info, freq, 0) #At each frequency matrices Z and F are assembled, but only these both matrices, not the derivates

        U = solve_(list_Z[0], list_F[0]) #Solve Z * U = F
        
        Psol1.x.array[:offset] = U.array_r[:offset]
        Qsol1.x.array[:(len(U.array_r) - offset)] = U.array_r[offset:]

        Pav1[ii] = petsc.assemble.assemble_scalar(form(Psol1*ds(1))) # Integrate the pression over the piston area

    t2 = time.time()
    k_output = 2*np.pi*freqvec/c0
    Z_analytical = (1-2*special.jv(1,2*k_output*radius)/(2*k_output*radius) + 1j*2*special.struve(1,2*k_output*radius)/(2*k_output*radius)) # Z_analytical concerns the analytical value of the acoustic impedance devided by rho0 * c0
    Z_center1 = 1j*k_output* Pav1 / surfarea #same here, for the numerical acoustic impedance divided by rho0 * c0

    file = geometry1+"_FOM_sidebox_" +str(side_box)
    with open(file+".txt", "w") as f: #Get the FOM data that can be used after
            f.write("% "+"Cubic FOM side box = "+str(side_box)+"\n")
            f.write("% "+"Frequency\tAnalytical\tNumerical")
            for xi in range(0, len(freqvec)):
                valeurs = "{}\t{}\t{}\n".format(freqvec[xi], Z_analytical[xi], Z_center1.real[xi])
                f.write(valeurs)
    ev_relative_err_ReZ1, ev_relative_err_ImZ1 = relative_errZ(freqvec, Z_analytical, Z_center1)
    print("err Re(Z1) = {}".format(ev_relative_err_ReZ1))
    print("err Im(Z1) = {}".format(ev_relative_err_ImZ1))
    print('For the Classical frequency sweep :'+str(t2-t1)+'sec')



WCAWE_1f0 = True #Let the 1 time used WCAWE be performed
if WCAWE_1f0:
    print('\n')
    N1 = 40 #Number of vector in the basis
    freq01 = 960
    print('---------------------------------------------------------------------------')
    print('\t\tB1p modelisation, WCAWE frequency sweep with N='+str(N1)+', f0 : '+str(freq01)+"Hz.")
    print('---------------------------------------------------------------------------')
    
    freqvec = np.arange(80, 2001, 20)
    Pav_WCAWE1 = np.zeros(freqvec.size,dtype=np.complex_)
    Psol1, Qsol1 = Function(P), Function(Q)
    offset = P.dofmap.index_map.size_local * P.dofmap.index_map_bs

    t1 = time.time()
    list_Z, list_F = z_f_matrices_b1p_withDiff(mesh_info, submesh_info, freq01, 2)
    
    Vn = basis_N_WCAWE(list_Z, list_F, freq01, N1)
    for ii in range(freqvec.size):
        freq = freqvec[ii]
        
        if freq%100==0:
            print('current freq : ')
            print(freq)

        list_Z, list_F = z_f_matrices_b1p_withDiff(mesh_info, submesh_info, freq, 0) #At each frequency matrices Z and F are assembled, but only these both matrices, not the derivates
        Zn, Fn = Zn_Fn_matrices(list_Z[0], list_F[0], Vn)
        alpha = solve_WCAWE(Zn, Fn)

        U = reduced_solution(Vn, alpha)
        Psol1, Qsol1 = Function(P), Function(Q)
        offset = P.dofmap.index_map.size_local * P.dofmap.index_map_bs
        Psol1.x.array[:offset] = U.array_r[:offset]
        Qsol1.x.array[:(len(U.array_r) - offset)] = U.array_r[offset:]

        Pav_WCAWE1[ii] = petsc.assemble.assemble_scalar(form(Psol1*ds(1)))
    
    t3 = time.time()
    k_output = 2*np.pi*freqvec/c0

    if from_data:
        file = bgt_order+"_" + geometry
        Z_fom = import_FOM_result(file)[1] #Take the numerical FOM data
        Z_center1_wcawe = 1j*k_output* Pav_WCAWE1 / surfarea #Numerical data obtained with the WCAWE methode

        fig, ax = plt.subplots(figsize=(10,10))
        ax.plot(freqvec,Z_fom, c = 'b', label=r'$\sigma_{FOM}$', lw = 2)
        ax.plot(freqvec,Z_center1_wcawe.real, c = 'r', label=r'$\sigma_{WCAWE}$', lw = 2)
        
        ax.legend()
        plt.xlabel('Frequency (Hz), f0 = '+str(freq01)+"Hz")
        plt.ylabel(r'$\sigma$')
        plt.title("WCAWE with 1 frequency, N = "+str(N1)+"vector")
        #plt.show()
        plt.savefig(file + '_WCAWE_1f0_N_'+str(N1)+'.png')

        title = bgt_order + " " + geometry
        with open(file + '_WCAWE_1f0_N_'+str(N1)+".txt", "w") as f: #Get the ROM data that can be used after
            f.write("% "+title+"\n")
            f.write("%"+ ' 1 time : f0 = '+str(freq01)+"Hz")
            f.write("% "+"\tFrequency\tFOM\tROM_WCAWE\n")
            for xi in range(0, len(freqvec)):
                valeurs = "{}\t{}\t{}\n".format(freqvec[xi], Z_fom[xi], Z_center1_wcawe.real[xi])
                f.write(valeurs)
        
        relative_err_ReZ1 = [0 for i in range(len(freqvec))]
        for ii in range(len(freqvec)):
            relative_err_ReZ1[ii] = abs(Z_center1_wcawe.real[ii]-Z_fom[ii])/Z_fom[ii]
        ev_relative_err_ReZ1 = sum(relative_err_ReZ1)/len(relative_err_ReZ1)
        print("err Re(Z1) = {}".format(ev_relative_err_ReZ1))


    else:
        Z_fom = Z_analytical
        file = geometry1+"_ROM_sidebox_" +str(side_box)
        Z_center1_wcawe = 1j*k_output * Pav_WCAWE1 / surfarea

        fig, ax = plt.subplots(figsize=(10,10))
        ax.plot(freqvec,Z_fom, c = 'b', label=r'$\sigma_{FOM}$', lw = 2)
        ax.plot(freqvec,Z_center1_wcawe.real, c = 'r', label=r'$\sigma_{WCAWE}$', lw = 2)
        
        ax.legend()
        plt.xlabel('Frequency (Hz), f0 = '+str(freq01)+"Hz")
        plt.ylabel(r'$\sigma$')
        plt.title("WCAWE with 1 frequency, N1 = "+str(N1)+" vectors")
        #plt.show()
        plt.savefig(file + '_WCAWE_1f0_N_'+str(N1)+'.png')

        title = bgt_order + " " + geometry
        with open(file + '_WCAWE_1f0_N_'+str(N1)+'.txt', "w") as f: #Get the ROM data that can be used after
            f.write("% "+title+"\n")
            f.write("%"+ ' 1 time : f0 = '+str(freq01)+"Hz")
            f.write("% "+"\tFrequency\tFOM\tROM_WCAWE\n")
            for xi in range(0, len(freqvec)):
                valeurs = "{}\t{}\t{}\n".format(freqvec[xi], Z_center1[xi], Z_center1_wcawe.real[xi])
                f.write(valeurs)
        ev_relative_err_ReZ1, ev_relative_err_ImZ1 = relative_errZ(freqvec, Z_center1, Z_center1_wcawe)
        print("err Re(Z1) = {}".format(ev_relative_err_ReZ1))
        print("err Im(Z1) = {}".format(ev_relative_err_ImZ1))

    print('Frequency sweep all inclued: '+str(t3-t1))


WCAWE_2f0 = True #Let the 2 times used WCAWE be performed
if WCAWE_2f0:
    print('\n')
    N1 = 15 #Number of vector in the basis
    print('---------------------------------------------------------------------------')
    print('\t\tB1p modelisation, WCAWE frequency sweep with N='+str(N1)+', 2f0 : f01 =520Hz, f02 = 1490Hz.')
    print('---------------------------------------------------------------------------')
    freq01 = 560
    freqvec = np.arange(80, 1040, 20)
    Pav_WCAWE1 = np.zeros(freqvec.size,dtype=np.complex_)
    Psol1, Qsol1 = Function(P), Function(Q)
    offset = P.dofmap.index_map.size_local * P.dofmap.index_map_bs

    t1 = time.time()
    list_Z, list_F = z_f_matrices_b1p_withDiff(mesh_info, submesh_info, freq01, 2)
    
    Vn = basis_N_WCAWE(list_Z, list_F, freq01, N1)
    for ii in range(freqvec.size):
        freq = freqvec[ii]
        
        if freq%100==0:
            print('current freq : ')
            print(freq)

        list_Z, list_F = z_f_matrices_b1p_withDiff(mesh_info, submesh_info, freq, 0) #At each frequency matrices Z and F are assembled, but only these both matrices, not the derivates
        Zn, Fn = Zn_Fn_matrices(list_Z[0], list_F[0], Vn)
        alpha = solve_WCAWE(Zn, Fn)

        U = reduced_solution(Vn, alpha)
        Psol1, Qsol1 = Function(P), Function(Q)
        offset = P.dofmap.index_map.size_local * P.dofmap.index_map_bs
        Psol1.x.array[:offset] = U.array_r[:offset]
        Qsol1.x.array[:(len(U.array_r) - offset)] = U.array_r[offset:]

        Pav_WCAWE1[ii] = petsc.assemble.assemble_scalar(form(Psol1*ds(1)))

    freq02 = 1520
    freqvec = np.arange(1040, 2001, 20)
    N2=30 #Number of vector in the basis
    Psol1, Qsol1 = Function(P), Function(Q)
    offset = P.dofmap.index_map.size_local * P.dofmap.index_map_bs
    Pav_WCAWE2 = np.zeros(freqvec.size,dtype=np.complex_)
    list_Z, list_F = z_f_matrices_b1p_withDiff(mesh_info, submesh_info, freq02, 2)
    Vn = basis_N_WCAWE(list_Z, list_F, freq02, N2)
    for ii in range(freqvec.size):
        freq = freqvec[ii]
        
        if freq%100==0:
            print('current freq : ')
            print(freq)

        list_Z, list_F = z_f_matrices_b1p_withDiff(mesh_info, submesh_info, freq, 0) #At each frequency matrices Z and F are assembled, but only these both matrices, not there derivates
        Zn, Fn = Zn_Fn_matrices(list_Z[0], list_F[0], Vn) 
        alpha = solve_WCAWE(Zn, Fn)

        U = reduced_solution(Vn, alpha)
        
        Psol1.x.array[:offset] = U.array_r[:offset]
        Qsol1.x.array[:(len(U.array_r) - offset)] = U.array_r[offset:]

        Pav_WCAWE2[ii] = petsc.assemble.assemble_scalar(form(Psol1*ds(1)))

    t3 = time.time()
    Pav_WCAWE = np.concatenate((Pav_WCAWE1,Pav_WCAWE2))
    freqvec = np.arange(80, 2001, 20)
    k_output = 2*np.pi*freqvec/c0


    if from_data:
        file = bgt_order+"_" + geometry
        Z_fom = import_FOM_result(file)[1] #Take the numerical FOM data
        Z_center1_wcawe = 1j*k_output* Pav_WCAWE / surfarea #Numerical data obtained with the WCAWE methode

        fig, ax = plt.subplots(figsize=(10,10))
        ax.plot(freqvec,Z_fom, c = 'b', label=r'$\sigma_{FOM}$', lw = 2)
        ax.plot(freqvec,Z_center1_wcawe.real, c = 'r', label=r'$\sigma_{WCAWE}$', lw = 2)
        
        ax.legend()
        plt.xlabel('Frequency (Hz), f01 = '+str(freq01)+"Hz, f02 = "+str(freq02)+"Hz")
        plt.ylabel(r'$\sigma$')
        plt.title("WCAWE with 2 frequencies, N1 = "+str(N1)+", N2 = "+str(N2)+' vectors')
        #plt.show()
        plt.savefig(file + '_WCAWE_2f0_N_'+str(N1)+'_'+str(N2)+'.png')

        title = bgt_order + " " + geometry
        with open(file + '_WCAWE_2f0_N_'+str(N1)+'_'+str(N2)+".txt", "w") as f: #Get the ROM data that can be used after
            f.write("% "+title+"\n")
            f.write("%"+ ' 2 times : f01 = '+str(freq01)+"Hz, f02 = "+str(freq02)+"Hz")
            f.write("% "+"\tFrequency\tFOM\tROM_WCAWE\n")
            for xi in range(0, len(freqvec)):
                valeurs = "{}\t{}\t{}\n".format(freqvec[xi], Z_fom[xi], Z_center1_wcawe.real[xi])
                f.write(valeurs)
        
        relative_err_ReZ1 = [0 for i in range(len(freqvec))]
        for ii in range(len(freqvec)):
            relative_err_ReZ1[ii] = abs(Z_center1_wcawe.real[ii]-Z_fom[ii])/Z_fom[ii]
        ev_relative_err_ReZ1 = sum(relative_err_ReZ1)/len(relative_err_ReZ1)
        print("err Re(Z1) = {}".format(ev_relative_err_ReZ1))


    else:
        Z_fom = Z_analytical
        file = geometry1+"_ROM_sidebox_" +str(side_box)
        Z_center1_wcawe = 1j*k_output * Pav_WCAWE / surfarea

        fig, ax = plt.subplots(figsize=(10,10))
        ax.plot(freqvec,Z_fom, c = 'b', label=r'$\sigma_{FOM}$', lw = 2)
        ax.plot(freqvec,Z_center1_wcawe.real, c = 'r', label=r'$\sigma_{WCAWE}$', lw = 2)
        
        ax.legend()
        plt.xlabel('Frequency (Hz), f01 = '+str(freq01)+"Hz, f02 = "+str(freq02)+"Hz")
        plt.ylabel(r'$\sigma$')
        plt.title("WCAWE with 2 frequencies, N1 = "+str(N1)+", N2 = "+str(N2)+'vectors')
        #plt.show()
        plt.savefig(file + '_WCAWE_2f0_N_'+str(N1)+'_'+str(N2)+'.png')

        title = bgt_order + " " + geometry
        with open(file + '_WCAWE_2f0_N_'+str(N1)+'_'+str(N2)+".txt", "w") as f: #Get the ROM data that can be used after
            f.write("% "+title+"\n")
            f.write("%"+ ' 2 times : f01 = '+str(freq01)+"Hz, f02 = "+str(freq02)+"Hz")
            f.write("% "+"\tFrequency\tFOM\tROM_WCAWE\n")
            for xi in range(0, len(freqvec)):
                valeurs = "{}\t{}\t{}\n".format(freqvec[xi], Z_center1[xi], Z_center1_wcawe.real[xi])
                f.write(valeurs)
        ev_relative_err_ReZ1, ev_relative_err_ImZ1 = relative_errZ(freqvec, Z_center1, Z_center1_wcawe)
        print("err Re(Z1) = {}".format(ev_relative_err_ReZ1))
        print("err Im(Z1) = {}".format(ev_relative_err_ImZ1))

    print('Frequency sweep all inclued: '+str(t3-t1))



