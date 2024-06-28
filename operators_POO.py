import numpy as np
from scipy import special
import scipy.linalg as la
from scipy.sparse import csr_matrix 
from scipy.io import savemat
from sympy import symbols, diff, lambdify
import sympy as sy
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import time

from abc import ABC, abstractmethod

import gmsh
from dolfinx import plot
from basix.ufl import element
from dolfinx.io import gmshio
import dolfinx.mesh as msh
from mpi4py import MPI
from dolfinx.fem import Function, functionspace, assemble, form, petsc, Constant, assemble_scalar
from ufl import (TestFunction, TrialFunction, TrialFunctions,
                 dx, grad, inner, Measure, variable, FacetNormal, CellNormal)
import ufl
import petsc4py
from petsc4py import PETSc
import slepc4py.SLEPc as SLEPc




rho0   = 1.21
c0     = 343.8
source = 1

fr = symbols('fr')
k0 = 2*np.pi*fr/c0

class Mesh:
    

    def __init__(self, deg, side_box, radius, lc, geometry, model_name = "no_name"):
        '''
        Constructor of the class Mesh. A Mesh is created from a function implemented in the module geometries. This class is not perfect, it only implements geometries of this case.
        input : 
            side_box   = float : 
            radius     = float : 
            lc         = float :
            geometry   = fonction :
            model_name = str : 

        output : 
            Mesh

        '''
        self.deg = deg
        
        mesh_info, submesh_info = geometry(side_box, radius, lc, model_name)

        self.mesh         = mesh_info[0]
        self.mesh_tags    = mesh_info[1]
        self.mesh_bc_tags = mesh_info[2]
        self.xref         = mesh_info[3]
    
        self.submesh          = submesh_info[0]
        self.entity_maps_mesh = submesh_info[1]


    def set_deg(self, deg):
        self.deg = deg
    
    def integral_mesure(self):
        '''
        This function gives access to the integral operator over the mesh and submesh in Mesh instance
        input :

        output :
            dx  = Measure : integral over the whole domain
            ds  = Measure : integral over the tagged surfaces
            dx1 = Measure : integral over the whole subdomain

        '''
        mesh         = self.mesh
        mesh_tags    = self.mesh_tags
        mesh_bc_tags = self.mesh_bc_tags
    
        submesh = self.submesh

        dx  = Measure("dx", domain=mesh, subdomain_data=mesh_tags)
        ds  = Measure("ds", domain=mesh, subdomain_data=mesh_bc_tags)
        dx1 = Measure("dx", domain=submesh)
        
        return dx, ds, dx1

    def fonction_spaces(self, family = "Lagrange"):
        '''
        This function provide fonction spaces needed in the FEM. They are spaces where the test and trial functions will be declared.
        input : 
            family = str : family of the element

        output : 
            P = FunctionSpace : fonction space where the fonctions living in the acoutic domain will be declared
            Q = FonctionSpace : fonction space where the fonctions living in the subdomain will be declared
        '''
        deg     = self.deg
        mesh    = self.mesh
        submesh = self.submesh
    
        P1 = element(family, mesh.basix_cell(), deg)
        P = functionspace(mesh, P1)


        if deg != 1:
            # This is linked to the fact on the subdomain, we will deal with the derivate of the function declared in the acoustic domain
            deg = deg - 1
        Q1 = element(family, submesh.basix_cell(), deg)
        Q = functionspace(submesh, Q1)
    
        return P, Q

    def angle_rotation(self, sub = False):
        mesh    = self.mesh
        submesh = self.submesh
        if sub :
            n = CellNormal(submesh)
        else :
            n = FacetNormal(mesh)

        x_x = ufl.as_vector([1, 0, 0])
        x_y = ufl.as_vector([0, 1, 0])
        x_z = ufl.as_vector([0, 0, 1])

        theta = ufl.acos(inner(x_x, n))
        phi   = ufl.acos(inner(x_y, n))
        psi   = ufl.acos(inner(x_z, n))

        return theta, phi, psi
        

    def rotation_matrix(self, sub = False):
        theta, phi, psi = self.angle_rotation(sub)
        
        cos_theta = ufl.cos(theta)
        sin_theta = ufl.sin(theta)
        cos_phi   = ufl.cos(phi)
        sin_phi   = ufl.sin(phi)
        cos_psi   = ufl.cos(psi)
        sin_psi   = ufl.sin(psi)
        
        rot_mat = ufl.as_matrix([[cos_psi*cos_phi, cos_psi*sin_phi*sin_theta - sin_psi*cos_theta, cos_psi*sin_phi*cos_theta+sin_psi*sin_theta],
                                 [sin_psi*cos_phi, sin_psi*sin_phi*sin_theta + cos_psi*cos_theta, sin_psi*sin_phi*cos_theta-cos_psi*sin_theta],
                                 [-sin_phi, cos_phi*sin_theta, cos_phi*cos_theta]])
        return rot_mat

    

class Simulation:

    def __init__(self, mesh, operator, loading):
        '''
        Constructor of the class Simulation. 
        input : 
            mesh     = Mesh
            operator = Operator
            loading  = Loading
        output :
            Simulation
        '''
        self.mesh     = mesh
        self.operator = operator
        self.loading  = loading

    def set_mesh(self, mesh):
        '''
        Setter to change the geometry on the one the same simulation will be run
        input : 
            mesh = Mesh : new Mesh obtained from a new geometry
        '''
        self.mesh = mesh

    def set_operator(self, ope):
        '''
        Setter to change the operator applied on the simulation
        input : 
            ope = Operator 
        '''
        self.operator = ope

    def set_loading(self, loading):
        '''
        Setter to change the loading applied on the simulation
        input : 
            loading = Loading 
        '''
        self.loading = loading
    
    def FOM(self, freq, frequency_sweep = True):

        if frequency_sweep and not(isinstance(freq, int)):
            print('Frequency sweep')
            return self.freq_sweep_FOM(freq)
        elif isinstance(freq, int) and not(frequency_sweep):
            print('Singular frequency')
            return self.singular_frequency_FOM(freq)
    
    def freq_sweep_FOM(self, freqvec):
        '''
        This function runs a frequency sweep on the simulation to obtain the acoustic pressure along the vibrating surface.
        input : 
            freqvec = np.arange() : frequency interval

        output :
            Pav1 = np.array() : pressure field obtained along the vibrating plate
            
        '''
        ope     = self.operator
        loading = self.loading
        Pav1    = np.zeros(freqvec.size, dtype=np.complex_)
        mesh    = self.mesh

        list_coeff_Z_j = ope.deriv_coeff_Z(0)
        list_coeff_F_j = loading.deriv_coeff_F(0)
        
        P, Q         = mesh.fonction_spaces()
        Psol1, Qsol1 = Function(P), Function(Q)
        offset       = P.dofmap.index_map.size_local * P.dofmap.index_map_bs # This is a bit obscur so far
        
        _, ds, _ = mesh.integral_mesure()
        for ii in tqdm(range(freqvec.size)):
            freq = freqvec[ii]
            
            Z = ope.dZj(freq, list_coeff_Z_j[0])
            F = loading.dFj(freq, list_coeff_F_j[0])
            if ii == 0:
                print(f"Size of the global matrix: {Z.getSize()}")
                
            # Solve
            ksp = PETSc.KSP().create()
            ksp.setOperators(Z)
            ksp.setType("gmres") # Solver type 
            ksp.getPC().setType("lu") # Preconditionner type
            ksp.getPC().setFactorSolverType("mumps") # Various type of previous objects are available, and different tests have to be performed to find the best. Normaly this configuration provides best results

            X = F.copy()
            ksp.solve(F, X) # Inversion of the matrix
        
            Psol1.x.array[:offset] = X.array_r[:offset]
            Qsol1.x.array[:(len(X.array_r) - offset)] = X.array_r[offset:]
        
            Pav1[ii] = assemble_scalar(form(Psol1*ds(1)))
            ksp.destroy()
            X.destroy()
            Z.destroy()
            F.destroy()
            
        return Pav1

    def singular_frequency_FOM(self, freq):
        '''
        This function runs a frequency sweep on the simulation to obtain the acoustic pressure along the vibrating surface.
        input : 
            freq = int : frequency 

        output :
            Psol1 = Function : pressure field in the whole acoustic domain
            Qsol1 = Function : derivate of pressure field along the boundaries
            
        '''
        ope     = self.operator
        loading = self.loading
        mesh    = self.mesh

        list_coeff_Z_j = ope.deriv_coeff_Z(0)
        list_coeff_F_j = loading.deriv_coeff_F(0)
        
        P, Q         = mesh.fonction_spaces()
        Psol1, Qsol1 = Function(P), Function(Q)
        offset       = P.dofmap.index_map.size_local * P.dofmap.index_map_bs # This is a bit obscur so far
        
        _, ds, _ = mesh.integral_mesure()
        
        Z = ope.dZj(freq, list_coeff_Z_j[0])
        F = loading.dFj(freq, list_coeff_F_j[0])
        
        # Solve
        ksp = PETSc.KSP().create()
        ksp.setOperators(Z)
        ksp.setType("gmres") # Solver type 
        ksp.getPC().setType("lu") # Preconditionner type
        ksp.getPC().setFactorSolverType("mumps") # Various type of previous objects are available, and different tests have to be performed to find the best. Normaly this configuration provides best results

        X = F.copy()
        ksp.solve(F, X) # Inversion of the matrix
    
        Psol1.x.array[:offset] = X.array_r[:offset]
        Qsol1.x.array[:(len(X.array_r) - offset)] = X.array_r[offset:]
    
        ksp.destroy()
        X.destroy()
        Z.destroy()
        F.destroy()
            
        return Psol1, Qsol1
    
    def wcawe(self, N, freq):
        '''
        One of the most complex function. This function implements the WCAWE model order reduction method.
        input :
            N    = int : nb of vector in the projection basis
            freq = int : interpolation point

        output : 
            Vn = PETScMat : projection basis
        '''

        ope              = self.operator
        list_Z           = ope.list_Z
        list_coeff_Z     = ope.list_coeff_Z
        loading          = self.loading
        list_F           = self.loading.list_F
        list_coeff_F     = self.loading.list_coeff_F
        mesh             = self.mesh
        entity_maps_mesh = mesh.entity_maps_mesh
        d_jZ = ope.deriv_coeff_Z(N)     # All the derivates of the Global matrix will be needed, they are computed here
        d_jF = loading.deriv_coeff_F(N) # All the derivates of the froce vector will be needed, they are computed here

        # The following lines assembled the 0th and the 1st derivates of the global matrix, and the 0th derivate of the force vector
        # evaluated at the interpolation frequency
        # The global matrix and its first derivate are needed to construct each vector, that's why they are computed
        # outside of the loop
        Z_0 = ope.dZj(freq, d_jZ[0])      
        F_0 = loading.dFj(freq, d_jF[0]) 
        Z_1 = ope.dZj(freq, d_jZ[1])      
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
        v1 = F_0.copy()
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
                # j is a reality index and represents the index in the sum
                
                F_j = loading.dFj(freq, d_jF[j])
                
                Pq_1        = P_Q_w(Q, n, j, 1)
                P_q_1_value = Pq_1.getValue(0, n - j - 1)

                # First sum
                rhs1 = rhs1 + P_q_1_value*F_j
                
                Pq_1.destroy()
                F_j.destroy()
                if j > 1:
                    # The second sum starts only at j = 2
                    P_q_2        = P_Q_w(Q, n, i, 2)
                    P_q_2_values = P_q_2.getColumnVector(n-i-1)

                    Z_j = ope.dZj(freq, d_jZ[j])
                    
                    row_is = PETSc.IS().createStride(Vn.getSize()[0], first=0, step=1)
                    col_is = PETSc.IS().createStride(n-i, first=0, step=1)
                    
                    Vn_i       = Vn.createSubMatrix(row_is, col_is)
                    Vn_i       = Z_j.matMult(Vn_i) # Vn_i = Z_i * Vn_i
                    Vn_i_P_q_2 = Vn_i.createVecLeft()
                    Vn_i.mult(P_q_2_values, Vn_i_P_q_2)

                    # Second sum
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
                    Q.setValue(i, i, norm_vn) # Carefull it will be the place i+1. Q.setValue(2,3,7) will put 7 at the place (3,4)
                else:
                    v_i = Vn.getColumnVector(i)     # Careful, asking for the vector i will give the (i+1)th reality vector
                    Q.setValue(i, n-1, vn.dot(v_i)) # Carefull the function vn.dot(v_i) does the scalar product between vn and the conjugate of v_i
                    v_i.destroy()
    
            ## Gram-schmidt
            for i in range(n):
                v_i = Vn.getColumnVector(i)
                vn  = vn - vn.dot(v_i) * v_i
                v_i.destroy()
            vn.normalize()
            Vn.setValues([i for i in range(size_v1)], n-1, vn, PETSc.InsertMode.INSERT_VALUES) # Careful, setValues(ni, nj, nk) considers indices as indexed from 0. Vn.setValues([2,4,9], [4,5], [[10, 11],[20, 21], [31,30]]) will change values at (3,5) = 10, (3, 6) = 11, (5, 5) = 20 ... #vn has been computed, to append it at the nth place in the base, we set up the (n-1)th column
        ksp.destroy()
        
        Vn.assemble()
    
        return Vn

    def merged_WCAWE(self, list_N, list_freq):
        if len(list_N) != len(list_freq):
            print(f"WARNING : The list of nb vector values and the list of interpolated frequencies does not match: {len(list_N)} - {len(list_freq)}")

        size_Vn = sum(list_N)
        V1 = self.wcawe(list_N[0], list_freq[0])
        size_V1 = V1.getSize()[0]
        Vn = PETSc.Mat().create()
        Vn.setSizes((size_V1, size_Vn))
        Vn.setType("seqdense")  
        Vn.setFromOptions()
        Vn.setUp()    
        for i in range(V1.getSize()[0]):
            for j in range(V1.getSize()[1]):
                Vn[i, j] = V1[i, j]
        count = list_N[0]
        for i in range (1,len(list_freq)):
            Vi = self.wcawe(list_N[i], list_freq[i])
            for ii in range(Vi.getSize()[0]):
                for jj in range(Vi.getSize()[1]):
                    Vn[ii, count + jj] = Vi[ii, jj]
            count += list_N[i]
        Vn.assemble()
        return Vn
        
            

    def moment_matching_MOR(self, Vn, freqvec):
        '''
        This function does a frequency sweep of the reduced model where a moment matching method has been applied. It is basically the same function as FOM(), but two lines has been added to project the matrix and the vector force
        input :
            Vn      = PETScMatType : projection basis
            freqvec = np.arange() : frequency interval
        
        output :
            Pav1 = np.array() : pressure field obtained along the vibrating plate    
        '''
          
        ope     = self.operator
        loading = self.loading
        Pav1    = np.zeros(freqvec.size, dtype=np.complex_)
        mesh_   = self.mesh

        list_coeff_Z_j = ope.deriv_coeff_Z(0)
        list_coeff_F_j = loading.deriv_coeff_F(0)
        
        P, Q         = mesh_.fonction_spaces()
        Psol1, Qsol1 = Function(P), Function(Q)
        offset       = P.dofmap.index_map.size_local * P.dofmap.index_map_bs
         
        _, ds, _ = mesh_.integral_mesure()
        
        for ii in tqdm(range(freqvec.size)):
            freq   = freqvec[ii]
            Z      = ope.dZj(freq, list_coeff_Z_j[0])
            F      = loading.dFj(freq, list_coeff_F_j[0])
            Zn, Fn = Zn_Fn_matrices(Z, F, Vn) # Reduction of Z and F
            Zn.convert("seqaij")              # Conversion of Zn to be solved with superlu and mumps
            
            # Solve
            ksp = PETSc.KSP().create()
            ksp.setOperators(Zn)
            ksp.setType("gmres")                       # Solver type 
            ksp.getPC().setType("lu")                  # Preconditionner type
            ksp.getPC().setFactorSolverType("superlu") # Various type of previous objects are available, and different tests hae to be performed to find the best. Normaly this configuration provides best results
            
            alpha = Fn.copy()
            ksp.solve(Fn, alpha) # Inversion of the matrix
        
            X = F.copy()
            Vn.mult(alpha, X) # Projection back to the gloabl solution
            
            
            Psol1.x.array[:offset]                    = X.array_r[:offset]
            Qsol1.x.array[:(len(X.array_r) - offset)] = X.array_r[offset:]
            # Qsol1 can provide information in the q unknown which is the normal derivative of the pressure field along the boundary. Qsol1 is not used in this contribution.
        
        
            Pav1[ii] = assemble_scalar(form(Psol1*ds(1)))
            ksp.destroy()
            X.destroy()
            Z.destroy()
            F.destroy()

        return Pav1


    def compute_radiation_factor(self, freqvec, Pav):
        _, ds, _ = self.mesh.integral_mesure()
        surfarea = assemble_scalar(form(1*ds(1)))
        k_output = 2*np.pi*freqvec/c0
        Z_center = 1j*k_output* Pav / surfarea
        return Z_center

    
    def plot_radiation_factor(self, ax, freqvec, Pav, s = '', save_fig = False):
        #_, ds, _ = self.mesh.integral_mesure()
        #surfarea = petsc.assemble.assemble_scalar(form(1*ds(1)))
        #k_output = 2*np.pi*freqvec/c0
        #Z_center = 1j*k_output* Pav / surfarea
        Z_center = self.compute_radiation_factor(freqvec, Pav)
        if s == 'FOM_b1p':
            ax.plot(freqvec, Z_center.real, label = r'$\sigma_{b1p}$', c = 'green')
        elif s == 'FOM_b2p':
            ax.plot(freqvec, Z_center.real, label = r'$\sigma_{b2p}$', c = 'red')
        elif s == 'WCAWE':
            ax.plot(freqvec, Z_center.real, label = r'$\sigma_{WCAWE}$', c = 'red')
            
        else :
            ax.plot(freqvec, Z_center.real, label = s)
        ax.grid(True)
        ax.legend(loc='upper left')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(r'$\sigma$')
        if save_fig :
            plt.savefig("test.png")

    
        
       

class Operator(ABC):
    '''
    This method aims at being an abstract one. One will definied an implemented operator, and never use the followong constructor.
    '''
    def __init__(self, mesh):
        '''
        Constructor of the class Operator. The objective is to overwrite this constructor in order for the user to only use designed Operator.
        input : 
            mesh = Mesh : instance of the class Mesh, where the Operator is applied on

        output :
            Operator
        '''

        self.mesh        = mesh
        self.deg         = None
        self.list_Z      = None
        self.list_coeffZ = None

    @abstractmethod
    def dZj(self):
        '''
        This method will be definied for all implemented operator
        '''
        pass

    def deriv_coeff_Z(self, j):
        '''
        Frist compute all the derivates from the 0th to the jth one, of the coefficient in the global matrix.
        Secondly turn the coeff as lambda function
        input:
            j = int : the highest needed frequency
    
        output : 
            d_jZ = List[List[lamdaFunction]] : List of a List of the derivated coeff as lambda function w.r.t frequency 
        '''
        list_coeff_Z = self.list_coeff_Z
        d_jZexpr     = [[] for i in range(j+1)]
        d_jZexpr[0]  = list_coeff_Z
    
        for i in range(1, len(d_jZexpr)):
            d_jZexpr[i] = np.array([diff(coeff, fr) for coeff in d_jZexpr[i-1]])
    
        d_jZ = [[lambdify(fr, d_jZexpr_j, 'numpy') for d_jZexpr_j in d_jZexpr[i]] for i in range(len(d_jZexpr))]
        return d_jZ

    @abstractmethod
    def import_matrix(self, freq):
        pass


class B1p(Operator):

    def __init__(self, mesh):
        '''
        Constructor of the b1p operator.
        '''
        super().__init__(mesh)
        self.deg                       = self.mesh.deg
        self.list_Z, self.list_coeff_Z = self.b1p()
        

    def b1p(self):
        '''
        Create all the constant Form of the b1p operator on the given mesh/simulation and all the coeff of the global matrix, all the constant Form of the force block vector and the related coeff
    
        input :
            mesh_info    = List[]
            submesh_info = List[]
    
        output :
            list_Z       = np.array(Form) : List of the constant matrices of the b1p operator
            list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
    
        '''
        
        mesh_tags    = self.mesh.mesh_tags
        mesh_bc_tags = self.mesh.mesh_bc_tags
        xref         = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
    
        #deg         = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1 = Function(Q)
        fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        
        g1 = inner(p, u)*ds(3)
        g2 = inner(fx1*p, u)*ds(3)
        e  = inner(q, u)*dx1
    
        list_Z       = np.array([k, m, c, g1, g2, e])
        list_coeff_Z = np.array([1, -k0**2, -1, 1j*k0, 1, 1])
    
        return list_Z, list_coeff_Z


    def dZj(self, freq, list_coeff_Z_j):
        '''
        Create and assemble the jth derivate of global matrix of the b1p operator. 
        input:
            freq           = int : Frequency where the coeff will be evaluated
            list_coeff_Z_j = List[lamdaFunction] : List of the jth derivate of the coeff as lambda function w.r.t frequency 
        
        output : 
            Z = PETSc_MatType : Assembled jth derivated global matrix at the given frequency
        '''
        list_Z = self.list_Z

        mesh             = self.mesh.mesh
        submesh             = self.mesh.submesh
        entity_maps_mesh = self.mesh.entity_maps_mesh

        # The following lines save the bug when a coefficient is equal to zero
        c_0 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[0](freq)))
        c_1 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[1](freq)))
        c_2 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[2](freq)))
        c_3 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[3](freq)))
        c_4 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[4](freq)))
        c_5 = Constant(submesh, PETSc.ScalarType(list_coeff_Z_j[5](freq)))
        
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

        return Z

    def import_matrix(self, freq):
        list_coeff_Z_j = self.deriv_coeff_Z(0)
        Z = self.dZj(freq, list_coeff_Z_j[0])
        ope_str = 'b1p'
        ai, aj, av = Z.getValuesCSR()
        Zsp = csr_matrix((av, aj, ai))
        savemat('Z_'+ope_str+'_ref.mat', {'Z'+ope_str:Zsp})
        

class B2p(Operator):

    def __init__(self, mesh):
        '''
        Constructor of the b2p operator.
        '''
        super().__init__(mesh)
        self.deg                       = self.deg
        self.list_Z, self.list_coeff_Z = self.b2p()
        

    def b2p(self):
        '''
        Create all the constant Form of the b2p operator on the given mesh/simulation and all the coeff of the global matrix, all the constant Form of the force block vector and the related coeff
    
        input :
            mesh_info    = List[]
            submesh_info = List[]
    
        output :
            list_Z       = np.array(Form) : List of the constant matrices of the b1p operator
            list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
    
        '''
        mesh         = self.mesh.mesh
        mesh_tags    = self.mesh.mesh_tags
        mesh_bc_tags = self.mesh.mesh_bc_tags
        xref         = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        
        #deg         = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1 = Function(Q)
        fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        
        dp  = inner(grad(p), n) # dp/dn = grad(p) * n
        ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        g1  = inner(ddp, u)*ds(3)
        g2  = inner(p, u)*ds(3)
        g3  = inner(fx1*p, u)*ds(3)
        g4  = inner(fx1**2*p, u)*ds(3)
        e1  = inner(fx1*q, u)*dx1
        e2  = inner(q, u)*dx1
    
        list_Z       = np.array([k,      m,  c, g1,     g2,    g3, g4, e1,    e2])
        list_coeff_Z = np.array([1, -k0**2, -1,  1, -k0**2, 4j*k0,  2,  4, 2j*k0])
    
        return list_Z, list_coeff_Z


    def dZj(self, freq, list_coeff_Z_j):
        '''
        Create and assemble the jth derivate of global matrix of the b1p operator. 
        input:
            freq           = int : Frequency where the coeff will be evaluated
            list_coeff_Z_j = List[lamdaFunction] : List of the jth derivate of the coeff as lambda function w.r.t frequency 
        
        output : 
            Z = PETSc_MatType : Assembled jth derivated global matrix at the given frequency
        '''
        list_Z = self.list_Z

        mesh             = self.mesh.mesh
        entity_maps_mesh = self.mesh.entity_maps_mesh
        submesh      = self.mesh.submesh

        # The following lines save the bug when a coefficient is equal to zero
        c_0 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[0](freq)))
        c_1 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[1](freq)))
        c_2 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[2](freq)))
        c_3 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[3](freq)))
        c_4 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[4](freq)))
        c_5 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[5](freq)))
        c_6 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[6](freq)))
        c_7 = Constant(submesh, PETSc.ScalarType(list_coeff_Z_j[7](freq)))
        c_8 = Constant(submesh, PETSc.ScalarType(list_coeff_Z_j[8](freq)))
        
        a_00 = c_0*list_Z[0] + c_1*list_Z[1]
        a_01 = c_2*list_Z[2]
        a_10 = c_3*list_Z[3] + c_4*list_Z[4] + c_5*list_Z[5] + c_6*list_Z[6]
        a_11 = c_7*list_Z[7] + c_8*list_Z[8]
    
        z_00 = form(a_00)
        z_01 = form(a_01, entity_maps=entity_maps_mesh)
        z_10 = form(a_10, entity_maps=entity_maps_mesh)
        z_11 = form(a_11)
    
        z = [[z_00, z_01],
            [z_10, z_11]]
    
        Z = petsc.assemble_matrix_block(z)
        Z.assemble()

        return Z

    def import_matrix(self, freq):
        list_coeff_Z_j = self.deriv_coeff_Z(0)
        Z = self.dZj(freq, list_coeff_Z_j[0])
        ope_str = 'b2p'
        ai, aj, av = Z.getValuesCSR()
        Zsp = csr_matrix((av, aj, ai))
        savemat('Z_'+ope_str+'_ref.mat', {'Z'+ope_str:Zsp})

class B2p_tang(Operator):

    def __init__(self, mesh):
        '''
        Constructor of the b2p operator.
        '''
        super().__init__(mesh)
        self.deg                       = self.deg
        self.list_Z, self.list_coeff_Z = self.b2p()
        

    def b2p(self):
        '''
        Create all the constant Form of the b2p operator on the given mesh/simulation and all the coeff of the global matrix, all the constant Form of the force block vector and the related coeff
    
        input :
            mesh_info    = List[]
            submesh_info = List[]
    
        output :
            list_Z       = np.array(Form) : List of the constant matrices of the b1p operator
            list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
    
        '''
        mesh         = self.mesh.mesh
        mesh_tags    = self.mesh.mesh_tags
        mesh_bc_tags = self.mesh.mesh_bc_tags
        xref         = self.mesh.xref
        submesh      = self.mesh.submesh
        
        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        ns               = CellNormal(submesh)
        #deg         = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 

        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1 = Function(Q)
        fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v) * ds(3)
        #dp  = tangential_proj(grad(p), n) # dp/dn = grad(p) * n
        #ddp = tangential_proj(grad(dp[0] + dp[1] + dp[2]), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        #ddp = tangential_proj(div(grad(p)), n)
        #rot_matrix = self.mesh.rotation_matrix()
        ddp  = ufl.as_vector([p.dx(0).dx(0), p.dx(1).dx(1), p.dx(2).dx(2)])
        
        #ddpt = rot_matrix*tangential_proj(ddp, n)
        ddpt = tangential_proj(ddp, n)
        #ddpt = ddp
        
        g1   = inner(ddpt[0] + ddpt[1]+ ddpt[2], u) * ds(3)
        #g1   = inner(ddpt[0] + ddpt[2] , u) * ds(3) #Actually if it works we can try with ddpt[0] that should be null
        #g1   = inner(ddpt[0] + ddpt[1] + ddpt[2] , u) * ds(3)
        
        g2  = inner(p, u) * ds(3)
        g3  = inner(fx1*p, u) * ds(3)
        g4  = inner(fx1**2*p, u) * ds(3)

        #dq  = tangential_proj(ufl.as_vector(grad(q)), ns)
        #dq  = grad(q)
        #e0  = inner(dq[0] + dq[1] + dq[2], u)*dx1
        #e0  = inner(dq, u)*dx1
        e1  = inner(fx1*q, u) * dx1
        e2  = inner(q, u) * dx1
    
        list_Z       = np.array([k,      m,  c, g1,       g2,    g3, g4, e1,    e2])
        list_coeff_Z = np.array([1, -k0**2, -1, -1, -2*k0**2, 4j*k0,  2,  4, 2j*k0])
    
        return list_Z, list_coeff_Z


    def dZj(self, freq, list_coeff_Z_j):
        '''
        Create and assemble the jth derivate of global matrix of the b1p operator. 
        input:
            freq           = int : Frequency where the coeff will be evaluated
            list_coeff_Z_j = List[lamdaFunction] : List of the jth derivate of the coeff as lambda function w.r.t frequency 
        
        output : 
            Z = PETSc_MatType : Assembled jth derivated global matrix at the given frequency
        '''
        list_Z = self.list_Z

        mesh             = self.mesh.mesh
        submesh             = self.mesh.submesh
        entity_maps_mesh = self.mesh.entity_maps_mesh

        # The following lines save the bug when a coefficient is equal to zero
        c_0 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[0](freq)))
        c_1 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[1](freq)))
        c_2 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[2](freq)))
        c_3 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[3](freq)))
        c_4 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[4](freq)))
        c_5 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[5](freq)))
        c_6 = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[6](freq)))
        c_7 = Constant(submesh, PETSc.ScalarType(list_coeff_Z_j[7](freq)))
        c_8 = Constant(submesh, PETSc.ScalarType(list_coeff_Z_j[8](freq)))
        
        a_00 = c_0*list_Z[0] + c_1*list_Z[1]
        a_01 = c_2*list_Z[2]
        a_10 = c_3*list_Z[3] + c_4*list_Z[4] + c_5*list_Z[5] + c_6*list_Z[6]
        a_11 = c_7*list_Z[7] + c_8*list_Z[8]
    
        z_00 = form(a_00)
        z_01 = form(a_01, entity_maps=entity_maps_mesh)
        z_10 = form(a_10, entity_maps=entity_maps_mesh)
        z_11 = form(a_11)
    
        z = [[z_00, z_01],
            [z_10, z_11]]
    
        Z = petsc.assemble_matrix_block(z)
        Z.assemble()

        return Z

    def import_matrix(self, freq):
        list_coeff_Z_j = self.deriv_coeff_Z(0)
        Z = self.dZj(freq, list_coeff_Z_j[0])
        ope_str = 'b2p'
        ai, aj, av = Z.getValuesCSR()
        Zsp = csr_matrix((av, aj, ai))
        savemat('Z_'+ope_str+'_ref.mat', {'Z'+ope_str:Zsp})


class B3p(Operator):

    def __init__(self, mesh):
        '''
        Constructor of the b2p operator.
        '''
        super().__init__(mesh)
        self.deg                       = self.deg
        self.list_Z, self.list_coeff_Z = self.b3p()
        

    def b3p(self):
        '''
        Create all the constant Form of the b2p operator on the given mesh/simulation and all the coeff of the global matrix, all the constant Form of the force block vector and the related coeff
    
        input :
            mesh_info    = List[]
            submesh_info = List[]
    
        output :
            list_Z       = np.array(Form) : List of the constant matrices of the b1p operator
            list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
    
        '''
        mesh         = self.mesh.mesh
        mesh_tags    = self.mesh.mesh_tags
        mesh_bc_tags = self.mesh.mesh_bc_tags
        xref         = self.mesh.xref

        submesh      = self.mesh.submesh
        #entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        ns               = CellNormal(submesh)
        #deg         = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
        
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1 = Function(Q)
        fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)

        ddp  = ufl.as_vector([p.dx(0).dx(0), p.dx(1).dx(1), p.dx(2).dx(2)])
        ddpt = tangential_proj(ddp, n)
        
        g1  = inner(fx1*(ddpt[0] + ddpt[1] + ddpt[2]) , u)*ds(3)
        g2  = inner(ddpt[0] + ddpt[1] + ddpt[2], u) * ds(3)
        
        g4  = inner(fx1**3*p, u)*ds(3)
        g5  = inner(fx1**2*p, u)*ds(3)
        g6  = inner(fx1*p, u)*ds(3)
        g7  = inner(p, u)*ds(3)

        ddq = ufl.as_vector([q.dx(0).dx(0), q.dx(1).dx(1), q.dx(2).dx(2)])
        ddqt = tangential_proj(ddq, ns)
        #ddqt = ddq
        e0  = inner(ddqt[0] + ddqt[1] + ddqt[2], u)*dx1
        e1  = inner(fx1**2*q, u)*dx1
        e2  = inner(fx1*q, u)*dx1
        e3  = inner(q, u)*dx1
        
        list_Z       = np.array([k,      m,  c, g1,     g2, g4,     g5,        g6,        g7, e0, e1,     e2,       e3])
        list_coeff_Z = np.array([1, -k0**2, -1, -9, -3j*k0,  6, 18j*k0, -18*k0**2, -4j*k0**3, -1, 18, 18j*k0, -4*k0**2])
        
        return list_Z, list_coeff_Z


    def dZj(self, freq, list_coeff_Z_j):
        '''
        Create and assemble the jth derivate of global matrix of the b3p operator. 
        input:
            freq           = int : Frequency where the coeff will be evaluated
            list_coeff_Z_j = List[lamdaFunction] : List of the jth derivate of the coeff as lambda function w.r.t frequency 
        
        output : 
            Z = PETSc_MatType : Assembled jth derivated global matrix at the given frequency
        '''
        list_Z = self.list_Z

        mesh             = self.mesh.mesh
        submesh             = self.mesh.submesh
        entity_maps_mesh = self.mesh.entity_maps_mesh

        # The following lines solve the bug when a coefficient is equal to zero
        c_0  = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[0](freq)))
        c_1  = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[1](freq)))
        c_2  = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[2](freq)))
        c_3  = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[3](freq)))
        c_4  = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[4](freq)))
        c_5  = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[5](freq)))
        c_6  = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[6](freq)))
        c_7  = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[7](freq)))
        c_8  = Constant(mesh, PETSc.ScalarType(list_coeff_Z_j[8](freq)))
        c_9  = Constant(submesh, PETSc.ScalarType(list_coeff_Z_j[9](freq)))
        c_10 = Constant(submesh, PETSc.ScalarType(list_coeff_Z_j[10](freq)))
        c_11 = Constant(submesh, PETSc.ScalarType(list_coeff_Z_j[11](freq)))
        c_12 = Constant(submesh, PETSc.ScalarType(list_coeff_Z_j[12](freq)))
        
        a_00 = c_0*list_Z[0] + c_1*list_Z[1]
        a_01 = c_2*list_Z[2]
        a_10 = c_3*list_Z[3] + c_4*list_Z[4] + c_5*list_Z[5] + c_6*list_Z[6] + c_7*list_Z[7] + c_8*list_Z[8] 
        a_11 = c_9*list_Z[9] + c_10*list_Z[10] + c_11*list_Z[11] + c_12*list_Z[12]
        
        z_00 = form(a_00)
        z_01 = form(a_01, entity_maps=entity_maps_mesh)
        z_10 = form(a_10, entity_maps=entity_maps_mesh)
        z_11 = form(a_11)
    
        z = [[z_00, z_01],
            [z_10, z_11]]
    
        Z = petsc.assemble_matrix_block(z)
        Z.assemble()

        return Z  

    def import_matrix(self, freq):
        list_coeff_Z_j = self.deriv_coeff_Z(0)
        Z = self.dZj(freq, list_coeff_Z_j[0])
        ope_str = 'b3p'
        ai, aj, av = Z.getValuesCSR()
        Zsp = csr_matrix((av, aj, ai))
        savemat('Z_'+ope_str+'_ref.mat', {'Z'+ope_str:Zsp})

class Loading:
    '''
    A Loading is a force applied on the mesh. So far this class only implement the vibrating plate, but this class could be implemented as the abstract class Operator
    '''

    def __init__(self, mesh):
        '''
        Constructor of the class Loading
        '''
        self.mesh                      = mesh
        self.list_F, self.list_coeff_F = self.f()


    def f(self):
        '''
        This function create the coefficients list of the vibration plate.
        input :

        output :
            list_F       = np.array() : list of the frequency constant vectors
            list_coeff_F = np.array() : list of the coeffient
        '''
        mesh         = self.mesh.mesh
        submesh         = self.mesh.submesh
        mesh_tags    = self.mesh.mesh_tags
        mesh_bc_tags = self.mesh.mesh_bc_tags
        xref         = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
    
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        v, u = TestFunction(P), TestFunction(Q)
        
        f    = inner(source, v) * ds(1)
        zero = inner(Constant(submesh, PETSc.ScalarType(0)), u) * dx1
        
        list_F       = np.array([f, zero])
        list_coeff_F = np.array([1, 0])
    
        return list_F, list_coeff_F


    def dFj(self, freq, list_coeff_F_j):
        '''
        Create and assemble the jth derivate of force vector of the vibrating plate loading. 
        input:
            freq           = int : Frequency where the coeff will be evaluated
            list_coeff_F_j = List of the coeff in the jth derivate of the force vector as lambda fct
        
        output : 
            F = PETSc_VecType : Assembled jth derivated force vector at the given frequency
        '''
        list_F = self.list_F

        mesh   = self.mesh.mesh
        submesh   = self.mesh.submesh
        
        c_0 = Constant(mesh, PETSc.ScalarType(list_coeff_F_j[0](freq)))
        c_1 = Constant(submesh, PETSc.ScalarType(list_coeff_F_j[1](freq)))

        f_0 = c_0*list_F[0]
        f_1 = c_1*list_F[1]
        
        f = [form(f_0), form(f_1)]

        F = petsc.assemble_vector_nest(f)

        return F

    def deriv_coeff_F(self, j):
        '''
        Frist compute all the derivates from the 0th to the jth one, of the coefficient in the force block vector.
        Secondly turn the coeff as lambda function
        input:
            list_coeff_F = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
    
        output : 
            d_jF = List[List[lamdaFunction]] : List of a List of the derivated coeff as lambda function w.r.t frequency 
        '''
        list_coeff_F = self.list_coeff_F
        d_jFexpr     = [[] for i in range(j+1)]
        d_jFexpr[0]  = list_coeff_F
    
        for i in range(1, len(d_jFexpr)):
            d_jFexpr[i] = np.array([diff(coeff, fr) for coeff in d_jFexpr[i-1]])
    
        
        d_jF = [[lambdify(fr, d_jFexpr_j, 'numpy') for d_jFexpr_j in d_jFexpr[i]] for i in range(len(d_jFexpr))]
        return d_jF



def sub_matrix(Q, start, end):
    '''
    This function is to obtain the sub matrix need for the correction term (P_q_w)
    intput :
        Q     = PETScMatType : the matrix where the norms and the scalar products are stored
        start = int : start index, reality index
        end   = int : end index, reality index

    output : 
        submatrix = np.array() : sub matrix, as a numpy matrix, because the size will remain low
    '''
     
    row_is    = PETSc.IS().createStride(end  - start + 1, first=start - 1, step=1)
    col_is    = PETSc.IS().createStride(end - start + 1, first=start - 1, step=1)
    submatrix = Q.createSubMatrix(row_is, col_is)

    row_is.destroy()
    col_is.destroy()

    submatrix = submatrix.getValues([i for i in range(end - start+1)], [i for i in range(end - start+1)])
    return submatrix
        
def P_Q_w(Q, alpha, beta, omega):
    '''
    Correction term function.
    input :
        Q     = PETScMatType : the matrix where the norms and the scalar products are stored
        alpha = int : reality value
        beta  = int : reality value
        omega = int : starting point of the product

    output :
        P_q_w = PETScMatType : correction term
    '''
    
    P_q = np.identity(alpha - beta) #create the identity matrix M*M with M = alpha - beta

    for t in range(omega, beta+1):
        sub_Q = sub_matrix(Q, t, alpha - beta + t - 1)
        sub_Q = np.linalg.inv(sub_Q)
        P_q   = np.dot(P_q, sub_Q)

    # The following lignes convert the result to a PETSc type
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
    '''
    This perform a reduction of the matrix and the vector through a given basis
    input :
        Z  = PETScMatType : assembled global matrix
        F  = PETScVecType : assembled force vector
        Vn = PETScMatType : orthonomalized basis

    output :
        Zn  = PETScMatType : reduced global matrix
        Fn  = PETScVecType : reduced force vector
        
    '''

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

def SVD_ortho(Vn):
    '''
    This function performs an orthogonalization of a basis with a singular value decomposition, with the power of SLEPc.
    input :
        Vn = PETScMatType : initial basis

    output :
        L = PETScMatType : result basis
    '''
    n   = Vn.getSize()[0]
    m   = Vn.getSize()[1]
    print(f'n : {n}')
    
    svd = SLEPc.SVD().create()
    svd.setOperator(Vn)
    #svd.setFromOptions()
    svd.setDimensions(m)
    svd.solve()
    
    nsv = svd.getConverged()
    print(f'svd : {nsv}')
    
    L = PETSc.Mat().createDense([n, nsv], comm=PETSc.COMM_WORLD)
    L.setUp()  
    
    for i in range(nsv):
        
        u = PETSc.Vec().createSeq(n)
        v = PETSc.Vec().createSeq(m)

        sigma = svd.getSingularTriplet(i, u, v)
        #print(sigma)

        for j in range(n):
            L[j, i] = u[j]
    
    L.assemble()

    return L

def get_cond_nb(Z):
    n   = Z.getSize()[0]
    m   = Z.getSize()[1]
    print(f'n : {n}')
    
    svd = SLEPc.SVD().create()
    svd.setOperator(Z)
    #svd.setFromOptions()
    svd.setDimensions(m)
    svd.solve()
    
    nsv = svd.getConverged()
    print(f'svd : {nsv}')
    Sigma = []
    for i in range(nsv):
        
        u = PETSc.Vec().createSeq(n)
        v = PETSc.Vec().createSeq(m)

        sigma = svd.getSingularTriplet(i, u, v)
        Sigma.append(sigma)
    return max(Sigma)
    
def SVD_ortho2(Vn):
    '''
    This function might be delete further. This function performs an orthogonalization of a basis with a singular value decomposition based on python computation. It converts the PETScMatType to a numpy array, does the computation, and gives back the orthogonalized matrix in the PETScMatType type.
    input :
        Vn = PETScMatType : initial basis

    output :
        V_petsc = PETScMatType : result basis
    '''
    Vn = Vn.getDenseArray()

    L, S, R = la.svd(Vn)
    
    print(f'len S : {len(S)}')
    print(S)
    L_star = L[:,0:len(S)]
    print(L_star.shape)
    V_n = L_star

    V_petsc = PETSc.Mat().create()
    V_petsc.setSizes((V_n.shape[0], V_n.shape[1]))
    V_petsc.setType('aij')  
    V_petsc.setUp()

    for i in range(V_n.shape[0]):
        for j in range(V_n.shape[1]):
            V_petsc[i, j] = V_n[i, j]
    V_petsc.assemble()

    return V_petsc


def check_ortho(Vn):
    '''
    This function plot the scalar product between 2 following vector inside a basis, to check if they are orthogonal one to each other.
    input :
        Vn = PETScMatType : the basis

    output :
    '''
    N = Vn.getSize()[1]
    for i in range(N-1):
        vec1 = Vn.getColumnVector(i)
        vec2 = Vn.getColumnVector(i+1)
        result = vec1.dot(vec2)
        print("vec"+str(i)+" . vec"+str(i+1)+" = "+str(result))
    
def store_results(s, freqvec, Pav):
    with open('FOM_data/'+s+'.txt', 'w') as fichier:    
        for i in range(len(freqvec)):        
            fichier.write('{}\t{}\n'.format(freqvec[i], Pav[i]))

def import_frequency_sweep(s):
    with open('FOM_data/'+s+".txt", "r") as f:
        freqvec = list()
        Pav     = list()
        for line in f:
            if "%" in line:
                continue
            data    = line.split()
            freqvec.append(data[0])
            Pav.append(data[1])
            freqvec = [float(element) for element in freqvec]
            Pav     = [complex(element) for element in Pav]
    
    freqvec = np.array(freqvec)
    Pav     = np.array(Pav)
    
    return freqvec, Pav


def compute_analytical_radiation_factor(freqvec, radius):
    k_output = 2*np.pi*freqvec/c0
    Z_analytical = (1-2*special.jv(1,2*k_output*radius)/(2*k_output*radius) + 1j*2*special.struve(1,2*k_output*radius)/(2*k_output*radius)) #The impedance is divided by rho * c0, it becames the radiation coefficient
    return Z_analytical

def plot_analytical_result_sigma(ax, freqvec, radius):
    Z_analytical = compute_analytical_radiation_factor(freqvec, radius)
    ax.plot(freqvec, Z_analytical.real, label = r'$\sigma_{ana}$', c = 'blue')
    ax.legend()

def import_COMSOL_result(s):
    s = "COMSOL_data/"+s+"_COMSOL_results.txt"
    with open(s, "r") as f:
        frequency = list()
        results = list()
        for line in f:
            if "%" in line:
                # on saute la ligne
                continue
            data = line.split()
            frequency.append(data[0])
            results.append(data[1])
            frequency = [float(element) for element in frequency]
            results = [float(element) for element in results]
    return frequency, results

def harry_plotter(space, sol, str_value, show_edges = True):

    u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(space)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data[str_value] = sol.x.array.real
    u_grid.set_active_scalars(str_value)
    u_plotter = pyvista.Plotter(notebook=True)
    u_plotter.add_mesh(u_grid, show_edges=show_edges)
    u_plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        u_plotter.show(jupyter_backend='client')

def tangential_proj(u, n):
    print(f"u : {u}")
    #print(f"u size: {u.ufl_shape[0]}")
    proj_u = (ufl.Identity(n.ufl_shape[0]) - ufl.outer(n, n)) * u
    print(f"proj_u : {proj_u}")
    return proj_u

def tensor_norm(proj_u):
    norm_proj = ufl.sqrt(proj_u[0]**2 + proj_u[1]**2 + proj_u[2]**2)
    
    return norm_proj


def least_square_err(freqvec1, Z_center1, freqvec2, Z_center2):
    
    if len(freqvec1) >=  len(freqvec2):
        check = set(freqvec2) <= set(freqvec1)
        freqvec_in = freqvec2
        freqvec_out = freqvec1
        Pav_in = Z_center2
        Pav_out = Z_center1
    else :
        check = set(freqvec1) <= set(freqvec2)
        freqvec_in = freqvec1
        freqvec_out = freqvec2
        Pav_in = Z_center1
        Pav_out = Z_center2
    if check :
        
        err = 0
        for i in range(len(freqvec_in)):
            err += (Pav_in[i] - Pav_out[np.where(freqvec_out == freqvec_in[i])[0][0]])**2
    
        return err
    else :
        print("something went wrong")
        return None