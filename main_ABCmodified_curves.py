# Modules importations
import numpy as np
from scipy import special
from geometries import cubic_domain, spherical_domain, half_cubic_domain, broken_cubic_domain
from postprocess import relative_errZ,import_FOM_result
from dolfinx.fem import (form, Function, FunctionSpace, petsc)
import petsc4py
from petsc4py import PETSc
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from operators_POO import (Mesh,
                        B1p, B2p, B3p, 
                        B2p_modified_r, B3p_modified_r,
                        Loading, 
                        Simulation, 
                        import_frequency_sweep, import_COMSOL_result, store_results, store_resultsv2, store_resultsv3, plot_analytical_result_sigma,
                        least_square_err, compute_analytical_radiation_factor)
print("test2")
# Choice of the geometry among provided ones
geometry1 = 'cubic'
geometry2 = 'large'
geometry  = geometry1 + '_'+ geometry2

if   geometry2 == 'small':
    side_box = 0.11
    lc       = 8e-3
elif geometry2 == 'large':
    side_box = 0.40
    lc       = 2e-2
else :
    print("Enter your own side_box and mesh size in the code")
    side_box = 0.40
    lc       = 1e-2 #Typical mesh size : Small case : 8e-3 Large case : 2e-3

if   geometry1 == 'cubic':
    geo_fct = cubic_domain
elif geometry1 == 'spherical':
    geo_fct = spherical_domain
elif geometry1 == 'half_cubic':
    geo_fct = half_cubic_domain
elif geometry1 == 'broken_cubic':
    geo_fct = broken_cubic_domain
else :
    print("WARNING : May you choose an implemented geometry")

# Simulation parameters
radius  = 0.1                               # Radius of the baffle
rho0    = 1.21                              # Density of the air
c0      = 343.8                             # Speed of sound in air
freqvec = np.arange(80, 2001, 20)           # List of the frequencies


if False:
    # Creation a simulation with new operator B2p
    dimP = 2
    if False :
        dimQ = dimP - 1
    else :
        dimQ = dimP

    mesh_   = Mesh(dimP, side_box, radius, lc, geo_fct)
    loading = Loading(mesh_)

    ope2    = B2p(mesh_)
    simu2   = Simulation(mesh_, ope2, loading)

    if from_data_b2p:
        s1 = 'FOM_b2p'
        s  = s1 + '_' + geometry
        freqvec2, PavFOM2 = import_frequency_sweep(s)
    else :
        freqvec2 = freqvec
        PavFOM2 = simu2.FOM(freqvec2)
        s1 = 'FOM_b2p'
        s  = s1 + '_' + geometry
        store_results(s, freqvec2, PavFOM2)

    ope2_modified    = B2p_modified_r(mesh_)
    simu2_modified   = Simulation(mesh_, ope2_modified, loading)

    if from_data_b2p_modified_r:
        s1 = 'FOM_b2p_modified_r'
        s  = s1 + '_' + geometry
        freqvec2, PavFOM2_modified = import_frequency_sweep(s)
    else :
        freqvec2_modified = freqvec
        PavFOM2_modified = simu2_modified.FOM(freqvec2_modified)
        s1 = 'FOM_b2p_modified_r'
        s  = s1 + '_' + geometry
        store_results(s, freqvec2, PavFOM2_modified)

    fig1, ax1 = plt.subplots(figsize=(16,9))
    simu2.plot_radiation_factor(ax1, freqvec2, PavFOM2, s = 'FOM_b2p')
    simu2_modified.plot_radiation_factor(ax1, freqvec2, PavFOM2_modified,  s = 'FOM_b2p_modified_r')
    ax1.set_ylim(0,2)
    plt.savefig("/root/WCAWE_POO_github/b2p_modif_r.png")


    ope2_modified    = B2p_modified_dp_dq(mesh_)
    simu2_modified   = Simulation(mesh_, ope2_modified, loading)

    if from_data_b2p_modified_dp_dq:
        s1 = 'FOM_b2p_modified_dp_dq'
        s  = s1 + '_' + geometry
        freqvec2, PavFOM2_modified = import_frequency_sweep(s)
    else :
        freqvec2_modified = freqvec
        PavFOM2_modified = simu2_modified.FOM(freqvec2_modified)
        s1 = 'FOM_b2p_modified__dp_dq'
        s  = s1 + '_' + geometry
        store_results(s, freqvec2, PavFOM2_modified)

    fig2, ax2 = plt.subplots(figsize=(16,9))
    simu2.plot_radiation_factor(ax2, freqvec2, PavFOM2, s = 'FOM_b2p')
    simu2_modified.plot_radiation_factor(ax2, freqvec2, PavFOM2_modified,  s = 'FOM_b2p_modified__dp_dq')
    ax2.set_ylim(0,2)
    plt.savefig("/root/WCAWE_POO_github/b2p_modif__dp_dq.png")

    # Creation a simulation with new operator B3p
    dimP = 3
    if False :
        dimQ = dimP - 1
    else :
        dimQ = dimP

    mesh_   = Mesh(dimP, side_box, radius, lc, geo_fct)
    loading = Loading(mesh_)

    ope3    = B3p(mesh_)
    simu3   = Simulation(mesh_, ope3, loading)

    if from_data_b3p:
        s1 = 'FOM_b3p'
        s  = s1 + '_' + geometry
        freqvec3, PavFOM3 = import_frequency_sweep(s)
    else :
        freqvec3 = freqvec
        PavFOM3 = simu3.FOM(freqvec3)
        s1 = 'FOM_b3p'
        s  = s1 + '_' + geometry
        store_results(s, freqvec3, PavFOM3)

    ope3_modified    = B3p_modified_r(mesh_)
    simu3_modified   = Simulation(mesh_, ope3_modified, loading)

    if from_data_b3p_modified_r:
        s1 = 'FOM_b3p_modified_r'
        s  = s1 + '_' + geometry
        freqvec3, PavFOM3_modified = import_frequency_sweep(s)
    else :
        freqvec3 = freqvec
        PavFOM3_modified = simu3_modified.FOM(freqvec3)
        s1 = 'FOM_b3p_modified_r'
        s  = s1 + '_' + geometry
        store_results(s, freqvec3, PavFOM3_modified)

    fig3, ax3 = plt.subplots(figsize=(16,9))
    simu3.plot_radiation_factor(ax3, freqvec3, PavFOM3, s = 'FOM_b3p')
    simu3_modified.plot_radiation_factor(ax3, freqvec3, PavFOM3_modified,  s = 'FOM_b3p_modified_r')
    ax3.set_ylim(0,2)
    plt.savefig("/root/WCAWE_POO_github/b3p_modif.png")


def main_ABCmodified_curves_fct(dimP, 
                                dimQ, 
                                str_ope, 
                                from_data, 
                                freqvec, 
                                geo_fct, 
                                ax, 
                                plot_row_columns = False,
                                plot_heatmap     = False,
                                plot_cond        = False,
                                plot_svlistZ     = False):
    
    mesh_    = Mesh(dimP, dimQ, side_box, radius, lc, geo_fct)
    loading = Loading(mesh_)

    if str_ope == "b1p":
        ope = B1p(mesh_)
    elif str_ope == "b2p":
        ope = B2p(mesh_)
    elif str_ope == "b2p_modified_r":
        ope = B2p_modified_r(mesh_)
    elif str_ope == "b3p":
        ope = B3p(mesh_)
    elif str_ope == "b3p_modified_r":
        ope = B3p_modified_r(mesh_)
    else:
        print("Operator doesn't exist")
        return

    simu    = Simulation(mesh_, ope, loading)
    if "modified" in str_ope:
        s1 = 'modified_ope/'
    else:
        s1 = 'tangential/tangential_'

    s2 = geometry + '_' + str_ope + '_' + str(lc) + '_' + str(dimP) + '_' + str(dimQ)
    s1 += s2

    if plot_row_columns:
        simu.plot_row_columns_norm(freq = 250, s = s2)
    
    if plot_heatmap:
        simu.plot_matrix_heatmap(freq = 250, s = s2)

    if plot_cond:
        simu.plot_cond(freqvec, s = s2)
    
    if plot_svlistZ:
        simu.plot_sv_listZ(s = s2)
    
    
    if from_data:
        freqvec, PavFOM = import_frequency_sweep(s1)
    else :
        freqvec = freqvec
        PavFOM = simu.FOM(freqvec)
        store_resultsv3(s1, freqvec, PavFOM, simu)
    
    simu.plot_radiation_factor(ax, freqvec, PavFOM, s = s1, compute = not(from_data))


##############################################
# The following is the plot of norms of the rows and cols, heatmap at 250Hz,
# conditioning curves, conditioning list 
# b1p with dimP = 1 | dimQ = 1
# b2p with dimP = 3 | dimQ = 3
# b3p with dimP = 4 | dimQ = 4

if False :

    fig, ax = plt.subplots(figsize=(16,9))
    
    dimP = 1
    str_ope = "b1p"
    from_data = True
    freqvec = np.arange(80, 2001, 20)  
    main_ABCmodified_curves_fct(dimP, 
                                str_ope, 
                                from_data, 
                                freqvec, 
                                geo_fct, 
                                ax, 
                                plot_row_columns = False,
                                plot_heatmap     = False,
                                plot_cond        = False,
                                plot_svlistZ     = False)

    dimP = 2
    str_ope = "b2p"
    from_data = True
    freqvec = np.arange(80, 2001, 20)  
    main_ABCmodified_curves_fct(dimP, 
                                str_ope, 
                                from_data, 
                                freqvec, 
                                geo_fct, 
                                ax, 
                                plot_row_columns = True,
                                plot_heatmap     = True,
                                plot_cond        = True,
                                plot_svlistZ     = True)

    dimP = 3
    str_ope = "b3p"
    from_data = True
    freqvec = np.arange(80, 2001, 20)  
    main_ABCmodified_curves_fct(dimP, 
                                str_ope, 
                                from_data, 
                                freqvec, 
                                geo_fct, 
                                ax, 
                                plot_row_columns = True,
                                plot_heatmap     = False,
                                plot_cond        = True,
                                plot_svlistZ     = False)
    
    plt.close()

##############################################
# The followinf compute the frequency sweep of the b2p_modified_r operator

if False:
    fig, ax = plt.subplots(figsize=(16,9))
    
    dimP = 3
    str_ope = "b2p_modified_r"
    from_data = False
    freqvec = np.arange(80, 2001, 20)  
    main_ABCmodified_curves_fct(dimP, 
                                str_ope, 
                                from_data, 
                                freqvec, 
                                geo_fct, 
                                ax, 
                                plot_row_columns = False,
                                plot_heatmap     = False,
                                plot_cond        = False,
                                plot_svlistZ     = False)

##############################################
# The following compute the frequency sweep of the b3p_modified_r operator

if False:
    fig, ax = plt.subplots(figsize=(16,9))
    plot_analytical_result_sigma(ax, freqvec, radius)
    dimP = 3
    str_ope = "b3p"
    from_data = False
    freqvec = np.arange(80, 2001, 20)  
    main_ABCmodified_curves_fct(dimP, 
                                str_ope, 
                                from_data, 
                                freqvec, 
                                geo_fct, 
                                ax, 
                                plot_row_columns = False,
                                plot_heatmap     = False,
                                plot_cond        = False,
                                plot_svlistZ     = False)

    ax.legend()
    ax.set_ylim(0,2)
    plt.savefig("/root/WCAWE_POO_tangential-main/curves/test.png")


##############################################
# The following is the downloading images for b3p_modified_r conditionning ppt

if False:
    fig, ax = plt.subplots(figsize=(16,9))

    dimP = 3
    str_ope = "b3p_modified_r"
    from_data = False
    freqvec = np.arange(80, 2001, 20)  
    main_ABCmodified_curves_fct(dimP, 
                                str_ope, 
                                from_data, 
                                freqvec, 
                                geo_fct, 
                                ax, 
                                plot_row_columns = False,
                                plot_heatmap     = False,
                                plot_cond        = False,
                                plot_svlistZ     = False)

    dimP = 2
    str_ope = "b2p_modified_r"
    from_data = False
    freqvec = np.arange(80, 2001, 20)  
    main_ABCmodified_curves_fct(dimP, 
                                str_ope, 
                                from_data, 
                                freqvec, 
                                geo_fct, 
                                ax, 
                                plot_row_columns = False,
                                plot_heatmap     = False,
                                plot_cond        = False,
                                plot_svlistZ     = False)


if True:
    fig, ax = plt.subplots(figsize=(16,9))

    dimP = 2
    dimQ = 2
    str_ope = "b2p_modified_r"
    from_data = False
    freqvec = np.arange(80, 2001, 20)  
    main_ABCmodified_curves_fct(dimP, dimQ, str_ope, from_data, freqvec, geo_fct, ax,
                                plot_row_columns = True,
                                plot_heatmap     = True,
                                plot_cond        = True,
                                plot_svlistZ     = True)

    dimP = 3
    dimQ = 3
    str_ope = "b3p_modified_r"
    from_data = False
    freqvec = np.arange(80, 2001, 20)  
    main_ABCmodified_curves_fct(dimP, dimQ, str_ope, from_data, freqvec, geo_fct, ax,
                                plot_row_columns = True,
                                plot_heatmap     = True,
                                plot_cond        = True,
                                plot_svlistZ     = True)

    plot_analytical_result_sigma(ax, freqvec, radius)

    ax.legend()
    ax.set_ylim(0,2)
    plt.tight_layout()
    plt.savefig("/root/WCAWE_POO_tangential-main/curves/test.png")