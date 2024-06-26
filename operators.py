import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import gmsh

from dolfinx.io import gmshio
import dolfinx.mesh as msh
from mpi4py import MPI
from dolfinx.fem import Function, FunctionSpace, assemble, form, petsc, Constant
from ufl import (TestFunction, TrialFunction, TrialFunctions,
                 dx, grad, inner, Measure, FacetNormal, diff, variable)
import petsc4py
from petsc4py import PETSc



rho0 = 1.21
c0 = 343.8
source = 1

def fonction_spaces(mesh_info, submesh_info, deg):
    mesh = mesh_info[0]
    submesh = submesh_info[0]
    P = FunctionSpace(mesh, ("Lagrange", deg))
    if deg != 1:
        deg = deg - 1
    Q = FunctionSpace(submesh, ("Lagrange", deg))
    return P, Q

def integral_mesure(mesh_info, submesh_info):
    mesh = mesh_info[0]
    mesh_tags = mesh_info[1]
    mesh_bc_tags = mesh_info[2]

    submesh = submesh_info[0]
    
    # Create measure for integral over boundary
    dx = Measure("dx", domain=mesh, subdomain_data=mesh_tags)
    ds = Measure("ds", domain=mesh, subdomain_data=mesh_bc_tags)
    dx1 = Measure("dx", domain=submesh)
    
    return dx, ds, dx1

def diff_j(k, freq, j):
    if j == 0:
        return k
    ### Recursive function that allowed to compute succesive derivates
    else:
        k = diff(k, freq) # k is the form to be derivated
        j = j - 1
        return diff_j(k, freq, j)

def z_f_matrices_b1p_withDiff(mesh_info, submesh_info, freq0, nb_d):
    ### One of the most important function
    
    mesh = mesh_info[0]
    mesh_tags = mesh_info[1]
    mesh_bc_tags = mesh_info[2]
    xref = mesh_info[3]

    submesh = submesh_info[0]
    entity_maps_mesh = submesh_info[1]
    
    list_Z = list()
    list_F = list()

    ### The BGT1 matrix will be built similarly as the operators used on the baffle_study folder
    deg = 1
    P, Q = fonction_spaces(mesh_info, submesh_info, deg)
    dx, ds, dx1 = integral_mesure(mesh_info, submesh_info) 

    # Define variational problem
    ############################
    p, q = TrialFunction(P), TrialFunction(Q)
    v, u = TestFunction(P), TestFunction(Q)

    fx1 = Function(Q)
    fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
    
    freq = Constant(mesh, PETSc.ScalarType(freq0))
    freq = variable(freq)
    k0 = 2*np.pi*freq/c0

    Gx = 1j*k0+fx1

    k = inner(grad(p), grad(v)) * dx
    m = inner(p, v) * dx
    c = inner(q, v)*ds(3)
    
    g = inner(Gx*p, u)*ds(3)
    d = inner(q, u)*dx1

    z_00 = form(k - k0**2*m)
    z_01 = form(-c, entity_maps=entity_maps_mesh)
    z_10 = form(g, entity_maps=entity_maps_mesh)
    z_11 = form(d)
    #a_00 = a_01
    z = [[z_00, z_01],
        [z_10, z_11]]


    f = [form(inner(source, v) * ds(1)), form(inner(Constant(mesh, PETSc.ScalarType(0)), u) * dx1)]
    Z = petsc.assemble_matrix_block(z)
    Z.assemble()

    F = petsc.assemble_vector_block(f, z)
    list_Z.append(Z)
    list_F.append(F)
    for j in range(1, nb_d+1):        
        ### Compute the needed derivates
        da_00 = form(diff_j(k - k0**2*m, freq, j))
        da_01 = form(diff_j(-c, freq, j), entity_maps=entity_maps_mesh)
        da_10 = form(diff_j(g, freq, j), entity_maps=entity_maps_mesh)
        da_11 = form(diff_j(d, freq, j))

        z_j = [[da_00, da_01],
            [da_10, da_11]]

        f_j = form([diff_j(inner(source, v) * ds(1), freq, j), diff_j(inner(Constant(mesh, PETSc.ScalarType(0)), u) * dx1, freq, j)])
        Z_j = petsc.assemble_matrix_block(z_j)
        Z_j.assemble()
        F_j = petsc.assemble_vector_block(f_j, z_j)
        list_Z.append(Z_j)
        list_F.append(F_j)
    ### The lists are the lists for the derivates of Z et F from order 0 to order nb_d
    return list_Z, list_F

def z_f_matrices_b2p_withDiff(mesh_info, submesh_info, freq0, nb_d):
    ### One of the most important function
    
    mesh = mesh_info[0]
    mesh_tags = mesh_info[1]
    mesh_bc_tags = mesh_info[2]
    xref = mesh_info[3]

    submesh = submesh_info[0]
    entity_maps_mesh = submesh_info[1]
    
    list_Z = list()
    list_F = list()

    deg = 2
    P = FunctionSpace(mesh, ("Lagrange", deg))
    Q = FunctionSpace(submesh, ("Lagrange", deg-1))

    # Define variational problem
    ############################
    p, q = TrialFunction(P), TrialFunction(Q)
    v, u = TestFunction(P), TestFunction(Q)

    dx, ds, dx1 = integral_mesure(mesh_info, submesh_info) 
    n = FacetNormal(mesh) # Normal to the boundaries
    
    fx = Function(Q)
    fx.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
    
    freq = Constant(mesh, PETSc.ScalarType(freq0))
    freq = variable(freq)
    k0 = 2*np.pi*freq/c0
  
    Fx = (2*1j*k0+4*fx)
    Hx = (2*fx**2-k0**2+4*1j*k0*fx)

    k = inner(grad(p), grad(v)) * dx # scalar product of the gradiants
    m = inner(p, v) * dx # dx represents the integral over the whole considered domain, here it is the whole acoustic domain
    c = inner(q, v)*ds(3) # ds represents the integral over a boundary, here the '3' one, which is the boundary where ABC is applied


    t = inner(Hx*p, u)*ds(3)
    dp = inner(grad(p), n) # dp/dn = grad(p) * n
    ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
    dt = inner(ddp, u)*ds(3)

    e = inner(Fx*q, u)*dx1 # q and u are defined in Q which is the function space defined on the boundary where ABC is applied as a whole surface domain

    z_00 = form(k - k0**2*m) # create square matrix
    z_01 = form(-c, entity_maps=entity_maps_mesh) # create rectangular matrix
    z_10 = form(t + dt, entity_maps=entity_maps_mesh)
    z_11 = form(e)
    #a_00 = a_01
    z = [[z_00, z_01],
        [z_10, z_11]]


    f = [form(inner(source, v) * ds(1)), form(inner(Constant(mesh, PETSc.ScalarType(0)), u) * dx1)]
    Z = petsc.assemble_matrix_block(z)
    Z.assemble()
    F = petsc.assemble_vector_block(f, z)
    list_Z.append(Z)
    list_F.append(F)
    for j in range(1, nb_d+1):
        ### Compute the needed derivates
        da_00 = form(diff_j(k - k0**2*m, freq, j))
        da_01 = form(diff_j(-c, freq, j), entity_maps=entity_maps_mesh)
        da_10 = form(diff_j(t + dt, freq, j), entity_maps=entity_maps_mesh)
        da_11 = form(diff_j(e, freq, j))

        z_j = [[da_00, da_01],
            [da_10, da_11]]

        f_j = form([diff_j(inner(source, v) * ds(1), freq, j), diff_j(inner(Constant(mesh, PETSc.ScalarType(0)), u) * dx1, freq, j)])
        Z_j = petsc.assemble_matrix_block(z_j)
        Z_j.assemble()
        F_j = petsc.assemble_vector_block(f_j, z_j)
        list_Z.append(Z_j)
        list_F.append(F_j)
    ### The lists are the lists for the derivates of Z et F from order 0 to order nb_d
    return list_Z, list_F

def basis_N_WCAWE(list_Z, list_F, freq0, N):
    ### The other most important function

    ### Create Q matrix
    Q = PETSc.Mat().create()
    Q.setSizes((N, N))  
    Q.setType("seqdense")  
    Q.setFromOptions()
    Q.setUp()       
    
    ### Obtain the first vector, its size will be needed to create the basis
    v1 = solve_(list_Z[0], list_F[0])
    norm_v1 = v1.norm()
    v1.normalize()
    Q.setValue(0, 0, norm_v1)
    size_v1 = v1.getSize()

    ### Create the empty basis
    Vn = PETSc.Mat().create()
    Vn.setSizes((size_v1, N))  
    Vn.setType("seqdense")  
    Vn.setFromOptions()
    Vn.setUp()    
    Vn.setValues([i for i in range(size_v1)], 0, v1, PETSc.InsertMode.INSERT_VALUES) #Vn[0] = v1

    ### Create the solver
    ksp = PETSc.KSP().create()
    ksp.setOperators(list_Z[0])
    ksp.setType("gmres")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("superlu") #superlu is better than mumps

    for n in range(2, N+1):
        rhs1 = list_Z[0].createVecLeft()
        
        if len(list_F) >= 1:
            #Compute the first sum. Instead of stoping the sum at n-1, it stops when the F_j matrix is null.
            #As it is all the derivates but not the Oth one that are present within the sum, 'if' condition checks if there are some derivates after the Oth one.

            for j in range(1, len(list_F)):   
                if not j >= n-1:              
                    Pq_1 = P_Q_w(Q, n, j, 1)
                    P_q_1_value = Pq_1.getValue(0, n - j - 1)
                    rhs1 = rhs1 + P_q_1_value*list_F[j]
        
        rhs2 = list_Z[0].createVecLeft()
        if len(list_Z)>1: #Check if there is derivate of Z
            vn_1 = Vn.getColumnVector(n-2)
            list_Z[1].mult(vn_1, rhs2) # rhs2 = Z_1 * vn_1

        rhs3 = list_Z[0].createVecLeft()
        if n>2 and len(list_Z) > 1:
            #As it is all the derivates but not the Oth one that are present within the sum, "len(list_Z)>1" checks if there are some derivates after the Oth one.
            #Compute the second sum. Instead of stoping the sum at n-1, it stops when the Z_j matrix is null.
            for i in range(2, len(list_Z)): 
                if not i >=n:
                    P_q_2 = P_Q_w(Q, n, i, 2)
                    P_q_2_values = P_q_2.getColumnVector(n-i-1)

                    row_is = PETSc.IS().createStride(Vn.getSize()[0], first=0, step=1)
                    col_is = PETSc.IS().createStride(n-i, first=0, step=1)

                    Vn_i = Vn.createSubMatrix(row_is, col_is)
                    Vn_i = list_Z[i].matMult(Vn_i) # Vn_i = Z_i * Vn_i
                    Vn_i_P_q_2 = Vn_i.createVecLeft()
                    Vn_i.mult(P_q_2_values, Vn_i_P_q_2)
                    
                    rhs3 = rhs3 + Vn_i_P_q_2
    
                    row_is.destroy()
                    col_is.destroy()
                    P_q_2.destroy()
                    P_q_2_values.destroy()
                    Vn_i.destroy()
                    Vn_i_P_q_2.destroy()         
        
        rhs = rhs1 - rhs2 - rhs3
        vn = Vn.createVecLeft()
        ksp.solve(rhs, vn)
        rhs.destroy()
        rhs1.destroy()
        rhs2.destroy()
        
        norm_vn = vn.norm()
        
        for i in range(n):
            if i == n-1:
                Q.setValue(i, i, norm_vn) #Carefull it will be the place i+1. Q.setValue(2,3,7) will put 7 at the place (3,4)
            else:
                v_i = Vn.getColumnVector(i) #Careful, asking for the vector i will give the (i+1)th vector
                Q.setValue(i, n-1, vn.dot(v_i)) #Carefull the function vn.dot(v_i) does the scalar product between vn and the conjugate of v_i
                v_i.destroy()

        ## Gram-schmidt
        for i in range(n):
            v_i = Vn.getColumnVector(i)
            vn = vn - vn.dot(v_i) * v_i
            v_i.destroy()
        vn.normalize()
        Vn.setValues([i for i in range(size_v1)], n-1, vn, PETSc.InsertMode.INSERT_VALUES) #Careful, setValues(ni, nj, nk) considers indices as indexed from 0. Vn.setValues([2,4,9], [4,5], [[10, 11],[20, 21], [31,30]]) will change values at (3,5) = 10, (3, 6) = 11, (5, 5) = 20 ... #vn has been computed, to append it at the nth place in the base, we set up the (n-1)th column
    ksp.destroy()
    
    Vn.assemble()
    

    return Vn

def Zn_Fn_matrices(Z, F, Vn):
    #print("I m in Zn_Fn_matrices and Z type is {}, F type is : {}, Vn type is {}.".format(Z.getType(), F.getType(), Vn.getType()))
    Vn_T = Vn.duplicate()
    Vn.copy(Vn_T)
    Vn_T.hermitianTranspose()
    Vn_T.assemble()

    Zn = PETSc.Mat().create()
    Zn.setSizes(Vn.getSizes()[1])  
    Zn.setType("seqdense")  
    Zn.setFromOptions()
    Zn.setUp()

    C = PETSc.Mat().create()
    C.setSizes(Vn.getSize()) 
    C.setType("seqdense")
    C.setFromOptions()
    C.setUp()

    Z.matMult(Vn, C) # C = Z * Vn
    Vn_T.matMult(C, Zn) # Zn = Vn_T * C = Vn_T * Z * Vn
    C.destroy()

    Fn = Zn.createVecLeft()
    Vn_T.mult(F, Fn) # Fn = Vn_T * F


    return Zn, Fn



def solve_(Z, F):
    # Solve
    #print("I m in solve_ and Z type is {}, F type is : {}.".format(Z.getType(), F.getType()))
    ksp2 = PETSc.KSP().create()
    ksp2.setOperators(Z)
    ksp2.setType("gmres")
    ksp2.getPC().setType("lu") 
    ksp2.getPC().setFactorSolverType("mumps") #mumps is better than superlu

    v = Z.createVecLeft()
    ksp2.solve(F, v)
    ksp2.destroy()

    return v

def solve_WCAWE(Zn, Fn):
    #print("I m in solve_WCAWE and Zn type is {}, Fn type is : {}.".format(Zn.getType(), Fn.getType()))
    Zn.convert("seqaij")
    ksp2 = PETSc.KSP().create()
    ksp2.setOperators(Zn)
    ksp2.setType("gmres")
    ksp2.getPC().setType("lu") 
    ksp2.getPC().setFactorSolverType("superlu") #mumps or superlu doesn't change anything

    alpha = Zn.createVecLeft()
    ksp2.solve(Fn, alpha)
    ksp2.destroy()
    Zn.convert("seqdense")

    return alpha

def reduced_solution(Vn, alpha):
    U = PETSc.Vec().create()
    U.setSizes(Vn.getSizes()[0])
    U.setType("seq")  
    U.setFromOptions()
    U.setUp()
    
    Vn.mult(alpha, U)
    return U

def sub_matrix(Q, start, end):

    row_is = PETSc.IS().createStride(end  - start + 1, first=start - 1, step=1)
    col_is = PETSc.IS().createStride(end - start + 1, first=start - 1, step=1)
    submatrix = Q.createSubMatrix(row_is, col_is)

    row_is.destroy()
    col_is.destroy()

    submatrix = submatrix.getValues([i for i in range(end - start+1)], [i for i in range(end - start+1)])
    return submatrix

def P_Q_w(Q, alpha, beta, omega):
    #print(alpha, beta)
    P_q = np.identity(alpha - beta) #create the identity matrix M*M with M = alpha - beta

    for t in range(omega, beta+1):
        sub_Q = sub_matrix(Q, t, alpha - beta + t - 1)
        sub_Q = np.linalg.inv(sub_Q)
        P_q = np.dot(P_q, sub_Q)
    
    P_q_w = PETSc.Mat().create()
    P_q_w.setSizes(P_q.shape, P_q.shape)
    P_q_w.setType("seqdense")  
    P_q_w.setFromOptions()
    P_q_w.setUp()

    for i in range(P_q.shape[0]):
        P_q_w.setValues(i, [j for j in range(P_q.shape[1])], P_q[i], PETSc.InsertMode.INSERT_VALUES)   
    P_q_w.assemble()
    return P_q_w
    
def check_ortho(Vn):
    N = Vn.getSize()[1]
    for i in range(N-1):
        vec1 = Vn.getColumnVector(i)
        vec2 = Vn.getColumnVector(i+1)
        result = vec1.dot(vec2)
        print("vec"+str(i)+" . vec"+str(i+1)+" = "+str(result))






        
