import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import gmsh

from dolfinx.io import gmshio
import dolfinx.mesh as msh
from mpi4py import MPI
from dolfinx.fem import Function, FunctionSpace, assemble, form, petsc, Constant
from ufl import (TestFunction, TrialFunction, TrialFunctions,
                 dx, grad, inner, Measure, FacetNormal, variable)
import petsc4py
from petsc4py import PETSc
from sympy import symbols, diff, lambdify
import sympy as sy


rho0 = 1.21
c0 = 343.8
source = 1

fr = symbols('fr')
k0 = 2*np.pi*fr/c0

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

def fonction_spaces(mesh_info, submesh_info, deg):
    mesh = mesh_info[0]
    submesh = submesh_info[0]
    P = FunctionSpace(mesh, ("Lagrange", deg))
    if deg != 1:
        deg = deg - 1
    Q = FunctionSpace(submesh, ("Lagrange", deg))
    return P, Q

def b1p(mesh_info, submesh_info):
    '''
    Create all the constant Form of the b1p operator on the given mesh/simulation and all the coeff of the global matrix, all the constant Form of the force block vector and the related coeff

    input :
        mesh_info = List[]
        submesh_info = List[]

    output :
        list_Z = np.array(Form) : List of the constant matrices of the b1p operator
        list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
        list_F = np.array(Form) : List of the constant vectors of the force block vector
        llist_coeff_F = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol

    '''
    mesh = mesh_info[0]
    mesh_tags = mesh_info[1]
    mesh_bc_tags = mesh_info[2]
    xref = mesh_info[3]

    submesh = submesh_info[0]
    entity_maps_mesh = submesh_info[1]

    deg = 1
    P, Q = fonction_spaces(mesh_info, submesh_info, deg)
    dx, ds, dx1 = integral_mesure(mesh_info, submesh_info) 

    p, q = TrialFunction(P), TrialFunction(Q)
    v, u = TestFunction(P), TestFunction(Q)
    
    fx1 = Function(Q)
    fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
    
    k = inner(grad(p), grad(v)) * dx
    m = inner(p, v) * dx
    c = inner(q, v)*ds(3)
    
    g1 = inner(p, u)*ds(3)
    g2 = inner(fx1*p, u)*ds(3)
    e = inner(q, u)*dx1

    list_Z = np.array([k, m, c, g1, g2, e])
    list_coeff_Z =  np.array([1, -k0**2, -1, 1j*k0, 1, 1])

    f = inner(source, v) * ds(1)
    zero = inner(Constant(mesh, PETSc.ScalarType(0)), u) * dx1
    
    list_F = np.array([f, zero])
    list_coeff_F = np.array([1, 0])

    return list_Z, list_coeff_Z, list_F, list_coeff_F

def dZj_b1p(freq, list_Z, list_coeff_Z_j, list_F, list_coeff_F_j, mesh, entity_maps_mesh):
    '''
    Create and assemble the jth derivate of global matrix of the b1p operator and the force vector at a given frequency. 
    input:
        freq = int : Frequency where the coeff will be evaluated
        list_Z = np.array(Form) : List of the constant matrices of the b1p operator
        list_coeff_Z_j = List of the coeff in the jth derivate of the Z matrix as lambda fct
        list_F = np.array(Form) : List of the constant vectors of the force block vector
        list_coeff_F_j = List of the coeff in the jth derivate of the force block vector as lambda fct
        entity_maps_mesh = dict : Dictionnary of facet for the product between non same dimension space function
    
    output : 
        Z = PETSc_MatType : Assembled jth derivated global matrix at the given frequency
        F = PETSc_VecType : Assembled jth derivated force block vector at the given frequency
    
    '''
    c_0 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[0](freq)))
    c_1 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[1](freq)))
    c_2 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[2](freq)))
    c_3 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[3](freq)))
    c_4 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[4](freq)))
    c_5 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[5](freq)))
    
    #a_00 = list_coeff_Z_j[0](freq)*list_Z[0] + list_coeff_Z_j[1](freq)*list_Z[1]
    #a_01 = list_coeff_Z_j[2](freq)*list_Z[2]
    #a_10 = list_coeff_Z_j[3](freq)*list_Z[3] + list_coeff_Z_j[4](freq)*list_Z[4]
    #a_11 = list_coeff_Z_j[5](freq)*list_Z[5]
    a_00 = c_0*list_Z[0] + c_1*list_Z[1]
    a_01 = c_2*list_Z[2]
    a_10 = c_3*list_Z[3] + c_4*list_Z[4]
    a_11 = c_5*list_Z[5]

    z_00 = form(a_00)
    z_01 = form(a_01, entity_maps=entity_maps_mesh)
    z_10 = form(a_10, entity_maps=entity_maps_mesh)
    z_11 = form(a_11)

    z = [[z_00, z_01],
        [z_10, z_11]]

    Z = petsc.assemble_matrix_block(z)
    Z.assemble()

    c_0 = Constant(mesh, PETSc.ScalarType(list_coeff_F_j[0](freq)))
    c_1 = Constant(mesh, PETSc.ScalarType(list_coeff_F_j[1](freq)))
    #f_0 = list_coeff_F_j[0](freq)*list_F[0]
    f_0 = c_0*list_F[0]
    f_1 = c_1*list_F[1]
    
    f = [form(f_0), form(f_1)]

    F = petsc.assemble_vector_block(f, z)
    #F = petsc.assemble_vector_nest(f)
    #F.assemble()
    return Z, F

def dZj(ope, freq, list_Z, list_coeff_Z_j, list_F, list_coeff_F_j, mesh, entity_maps_mesh):

    if ope == 'b1p':
        Z, F = dZj_b1p(freq, list_Z, list_coeff_Z_j, list_F, list_coeff_F_j, mesh, entity_maps_mesh)
    else:
        print('implement another operator')

    return Z, F

def deriv_coeff_Z(list_coeff_Z, j):
    '''
    Frist compute all the derivates from the 0th to the jth one, of the coefficient in the global matrix.
    Secondly turn the coeff as lambda function
    input:
        list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol

    output : 
        d_jZ = List[List[lamdaFunction]] : List of a List of the derivated coeff as lambda function w.r.t frequency 
    '''

    d_jZexpr = [[] for i in range(j+1)]
    d_jZexpr[0] = list_coeff_Z

    for i in range(1, len(d_jZexpr)):
        d_jZexpr[i] = np.array([diff(coeff, fr) for coeff in d_jZexpr[i-1]])

    d_jZ = [[lambdify(fr, d_jZexpr_j, 'numpy') for d_jZexpr_j in d_jZexpr[i]] for i in range(len(d_jZexpr))]
    return d_jZ

def deriv_coeff_F(list_coeff_F, j):
    '''
    Frist compute all the derivates from the 0th to the jth one, of the coefficient in the force block vector.
    Secondly turn the coeff as lambda function
    input:
        list_coeff_F = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol

    output : 
        d_jF = List[List[lamdaFunction]] : List of a List of the derivated coeff as lambda function w.r.t frequency 
    '''

    d_jFexpr = [[] for i in range(j+1)]
    d_jFexpr[0] = list_coeff_F

    for i in range(1, len(d_jFexpr)):
        d_jFexpr[i] = np.array([diff(coeff, fr) for coeff in d_jFexpr[i-1]])

    
    d_jF = [[lambdify(fr, d_jFexpr_j, 'numpy') for d_jFexpr_j in d_jFexpr[i]] for i in range(len(d_jFexpr))]
    return d_jF


def wcawe1(N, f_0, ope, list_Z, list_coeff_Z, list_F, list_coeff_F, mesh, entity_maps_mesh):
    
    d_jZ = deriv_coeff_Z(list_coeff_Z, N)
    d_jF = deriv_coeff_F(list_coeff_F, N)
    print(d_jZ[1])
    Z_0, F_0 = dZj(ope, f_0, list_Z, d_jZ[0], list_F, d_jF[0], mesh, entity_maps_mesh)
    Z_1, F_1 = dZj(ope, f_0, list_Z, d_jZ[1], list_F, d_jF[1], mesh, entity_maps_mesh)
    print('ok')

def wcawe(N, f_0, ope, list_Z, list_coeff_Z, list_F, list_coeff_F, mesh, entity_maps_mesh):
    '''

    input :
        N = int : nb of vector in the projection basis
        f_0 = int : interpolation point
        ope
        list_Z = np.array(Form) : List of the constant matrices of an operator
        ???list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
        list_F = np.array(Form) : List of the constant vectors of the force block vector
        entity_maps_mesh
        
        

    '''
    
    d_jZ = deriv_coeff_Z(list_coeff_Z, N)
    d_jF = deriv_coeff_F(list_coeff_F, N)

    Z_0, F_0 = dZj(ope, f_0, list_Z, d_jZ[0], list_F, d_jF[0], mesh, entity_maps_mesh)
    Z_1, _ = dZj(ope, f_0, list_Z, d_jZ[1], list_F, d_jF[1], mesh, entity_maps_mesh)

    ### Create the solver
    ksp = PETSc.KSP().create()
    ksp.setOperators(Z_0)
    ksp.setType("gmres")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("superlu")
    
    ### Create Q matrix
    Q = PETSc.Mat().create()
    Q.setSizes((N, N))  
    Q.setType("seqdense")  
    Q.setFromOptions()
    Q.setUp()       
    
    ### Obtain the first vector, its size will be needed to create the basis
    v1 = Z_0.createVecLeft()
    ksp.solve(F_0, v1)
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
    
    for n in range(2, N+1):
        rhs1 = Z_0.createVecLeft()
        rhs3 = Z_0.createVecLeft()

        for j in range(1, n):
            Z_j, F_j = dZj(ope, f_0, list_Z, d_jZ[j], list_F, d_jF[j], mesh, entity_maps_mesh)
            
            Pq_1 = P_Q_w(Q, n, j, 1)
            P_q_1_value = Pq_1.getValue(0, n - j - 1)
            rhs1 = rhs1 + P_q_1_value*F_j
            F_j.destroy()
            if j > 2:
                P_q_2 = P_Q_w(Q, n, i, 2)
                P_q_2_values = P_q_2.getColumnVector(n-i-1)

                row_is = PETSc.IS().createStride(Vn.getSize()[0], first=0, step=1)
                col_is = PETSc.IS().createStride(n-i, first=0, step=1)

                Vn_i = Vn.createSubMatrix(row_is, col_is)
                Vn_i = Z_j.matMult(Vn_i) # Vn_i = Z_i * Vn_i
                Vn_i_P_q_2 = Vn_i.createVecLeft()
                Vn_i.mult(P_q_2_values, Vn_i_P_q_2)
                
                rhs3 = rhs3 + Vn_i_P_q_2

                Z_j.destroy()
                row_is.destroy()
                col_is.destroy()
                P_q_2.destroy()
                P_q_2_values.destroy()
                Vn_i.destroy()
                Vn_i_P_q_2.destroy()
        
        rhs2 = Z_0.createVecLeft()

        vn_1 = Vn.getColumnVector(n-2)
        Z_1.mult(vn_1, rhs2) # rhs2 = Z_1 * vn_1        
        
        rhs = rhs1 - rhs2 - rhs3
        vn = Vn.createVecLeft()
        ksp.solve(rhs, vn)
        rhs.destroy()
        rhs1.destroy()
        rhs2.destroy()
        rhs3.destroy()
        
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







































