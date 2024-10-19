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
                        B1p, r_tang, B3p, 
                        Loading, 
                        Simulation, 
                        import_frequency_sweep, import_COMSOL_result, store_results, store_resultsv2, plot_analytical_result_sigma,
                        least_square_err, compute_analytical_radiation_factor)
print("test2")
# Choice of the geometry among provided ones
geometry1 = 'cubic'
geometry2 = 'small'
geometry  = geometry1 + '_'+ geometry2

if   geometry2 == 'small':
    side_box = 0.11
    lc       = 8e-3
elif geometry2 == 'large':
    side_box = 0.40
    lc       = 1e-2
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

# To compare to COMSOL
comsol_data = True'_'

if comsol_data:
    s = geometry
    frequency, results = import_COMSOL_result(s)

# Choice between using saved results or doing a new frequency sweep
from_data_b1p      = True
from_data_b2p_tang = True
from_data_b3p      = True 

b1p_plot_row_columns = False
b2p_plot_row_columns = True
b3p_plot_row_columns = True

b1p_plot_svd = False
b2p_plot_svd = True
b3p_plot_svd = True


#  Creation of a simulation with B1p
## Choice of a mesh, a loading, an operator -> Simulation
dimP = 3
if False :
    dimQ = dimP - 1
else :
    dimQ = dimP
mesh_   = Mesh(dimP, side_box, radius, lc, geo_fct)
loading = Loading(mesh_)
ope1    = B1p(mesh_)
simu1   = Simulation(mesh_, ope1, loading)

if from_data_b1p:
    s1 = 'FOM_b1p'
    s = s1 + '_' + geometry
    freqvec1, PavFOM1 = import_frequency_sweep(s)
else :
    freqvec1 = freqvec
    PavFOM1 = simu1.FOM(freqvec1)
    s1 = 'FOM_b1p'
    s = s1 + '_' + geometry
    store_results(s, freqvec1, PavFOM1)
    if True :
        list_s = [geometry1, geometry2, "b1p", str(lc), str(dimP), str(dimQ)]
        store_resultsv2(list_s, freqvec1, PavFOM1, simu1)

if b1p_plot_row_columns:
    simu1.plot_row_columns_norm(freq = 80, s = 'tang_b1p')
    #simu1.plot_matrix_heatmap(freq = 1000, s = 'tang_b1p')

if b1p_plot_svd:
    simu1.plot_cond(freqvec1, s ='tang_b1p_'+str(dimP)+'_'+str(dimQ))
    #simu1.plot_sv_listZ(s ='tang_b1p_')

# Creation a simulation with new operator B2ptang
dimP = 3
if False :
    dimQ = dimP - 1
else :
    dimQ = dimP

mesh_   = Mesh(dimP, side_box, radius, lc, geo_fct)
loading = Loading(mesh_)
ope2    = B2p_tang(mesh_)
simu2   = Simulation(mesh_, ope2, loading)

if from_data_b2p_tang:
    s1 = 'FOM_b2p'
    s  = s1 + '_' + geometry
    freqvec2, PavFOM2 = import_frequency_sweep(s)
else :
    freqvec2 = freqvec
    PavFOM2 = simu2.FOM(freqvec2)
    s1 = 'FOM_b2p'
    s  = s1 + '_' + geometry
    store_results(s, freqvec2, PavFOM2)
    if True :
        list_s = [geometry1, geometry2, "b2p", str(lc), str(dimP), str(dimQ)]
        store_resultsv2(list_s, freqvec2, PavFOM2, simu2)

if b2p_plot_row_columns:
    simu2.plot_row_columns_norm(freq = 80, s = 'tang_b2p')
    #simu2.plot_matrix_heatmap(freq = 1000, s = 'tang_b2p')

if b2p_plot_svd:
    simu2.plot_cond(freqvec2, s ='tang_b2p_'+str(dimP)+'_'+str(dimQ))
    #simu2.plot_sv_listZ(s ='tang_b2p_')

# Creation a simulation with new operator B3p
dimP = 3
if False :
    dimQ = dimP - 1
else :
    dimQ = dimP

mesh_   = Mesh(dimP, side_box, radius, lc, geo_fct)
ope3    = B3p(mesh_)
loading = Loading(mesh_)
simu3   = Simulation(mesh_, ope3, loading)

if from_data_b3p:
    s1 = 'FOM_b3p'
    s  = s1 + '_' + geometry
    freqvec3, PavFOM3 = import_frequency_sweep(s)
else :
    freqvec3 = freqvec
    PavFOM3 = simu3.FOM(freqvec2)
    s1 = 'FOM_b3p'
    s  = s1 + '_' + geometry
    store_results(s, freqvec3, PavFOM3)
    if True :
        list_s = [geometry1, geometry2, "b3p", str(lc), str(dimP), str(dimQ)]
        store_resultsv2(list_s, freqvec3, PavFOM3, simu3)

if b3p_plot_row_columns:
    simu3.plot_row_columns_norm(freq = 80, s = 'tang_b3p')
    #simu3.plot_matrix_heatmap(freq = 1000, s = 'b3p')

if b3p_plot_svd:
    simu3.plot_cond(freqvec3, s ='tang_b3p_'+str(dimP)+'_'+str(dimQ))
    #simu3.plot_sv_listZ(s ='tang_b3p_')

# Plot of the results with matplotlib - so far impossible except with jupyterlab
fig, ax = plt.subplots(figsize=(16,9))
simu1.plot_radiation_factor(ax, freqvec1, PavFOM1, s = 'FOM_b1p')
simu2.plot_radiation_factor(ax, freqvec2, PavFOM2,  s = 'FOM_b2p')
#print(f'PavFOM2 : {PavFOM2}')
run_B3p = True
if run_B3p:
    freqvec3 = freqvec
    simu3.plot_radiation_factor(ax, freqvec3, PavFOM3,  s = 'FOM_b3p')
if comsol_data:
    ax.plot(frequency, results, c = 'black', label=r'$\sigma_{COMSOL}$')
    ax.legend()

plot_analytical_result = True
if plot_analytical_result:
    plot_analytical_result_sigma(ax, freqvec, radius)
ax.set_ylim(0,2)
plt.savefig("test.png")

Z_ana = compute_analytical_radiation_factor(freqvec, radius)
err_B1p = least_square_err(freqvec, Z_ana.real, freqvec1, simu1.compute_radiation_factor(freqvec1, PavFOM1).real)
print(f'For lc = {lc} - L2_err(B1p) = {err_B1p}')

err_B2p = least_square_err(freqvec, Z_ana.real, freqvec2, simu2.compute_radiation_factor(freqvec2, PavFOM2).real)
print(f'For lc = {lc} - L2_err(B2p) = {err_B2p}')

if run_B3p:
    err_B3p = least_square_err(freqvec, Z_ana.real, freqvec3, simu3.compute_radiation_factor(freqvec3, PavFOM3).real)
    print(f'For lc = {lc} - L2_err(B3p) = {err_B3p}')