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


from operators_POO import Mesh, B1p, Loading, Simulation, import_frequency_sweep
print("test")
geometry1 = 'cubic'
geometry2 = 'small'
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
    lc       = 1e-2 #Typical mesh size : Small case : 8e-3 Large case : 2e-2

from_data_b1p = True
from_data_b2p = True

radius = 0.1

rho0 = 1.21
c0   = 343.8

freqvec = np.arange(80, 2001, 20)

from operators_POO import import_COMSOL_result

comsol_data = True

if comsol_data:
    s = geometry
    frequency, results = import_COMSOL_result(s)

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

mesh_   = Mesh(1, side_box, radius, lc, geo_fct)
loading = Loading(mesh_)


from operators_POO import B1p
ope1           = B1p(mesh_)
list_coeff_Z_j = ope1.deriv_coeff_Z(0)
simu1 = Simulation(mesh_, ope1, loading)

from operators_POO import store_results
#ope1.import_matrix(freq = 2000)
if from_data_b1p:
    s1 = 'FOM_b1p'
    s = s1 + '_' + geometry
    freqvec1, PavFOM1 = import_frequency_sweep(s)
else :
    PavFOM1 = simu1.FOM(freqvec)
    s1 = 'FOM_b1p'
    s = s1 + '_' + geometry
    freqvec1 = freqvec
    store_results(s, freqvec, PavFOM1)

from operators_POO import B2p

mesh_.set_deg(2)

ope2           = B2p(mesh_)
list_coeff_Z_j = ope2.deriv_coeff_Z(0)

loading        = Loading(mesh_)
list_coeff_F_j = loading.deriv_coeff_F(0)

simu2 = Simulation(mesh_, ope2, loading)


#ope2.import_matrix(freq = 2000)
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

from operators_POO import B2p_tang

mesh_.set_deg(2)

ope2spe   = B2p_tang(mesh_)
loading   = Loading(mesh_)
simu2spe  = Simulation(mesh_, ope2spe, loading)


#ope2.import_matrix(freq = 2000)
from_data_b2pspe = True
if from_data_b2pspe:
    s1 = 'FOM_b2pspe'
    s  = s1 + '_' + geometry
    freqvec2spe, PavFOM2spe = import_frequency_sweep(s)
else :
    freqvec2spe = freqvec
    PavFOM2spe = simu2spe.FOM(freqvec2spe)
    s1 = 'FOM_b2pspe'
    s  = s1 + '_' + geometry
    store_results(s, freqvec2spe, PavFOM2spe)


from operators_POO import B3p

mesh_.set_deg(3)

ope3    = B3p(mesh_)
loading = Loading(mesh_)

simu3   = Simulation(mesh_, ope3, loading)
freqvec3 = freqvec
PavFOM3 = simu3.FOM(freqvec3)


from operators_POO import plot_analytical_result_sigma

fig, ax = plt.subplots(figsize=(16,9))
simu1.plot_radiation_factor(ax, freqvec1, PavFOM1, s = 'FOM_b1p')
simu2.plot_radiation_factor(ax, freqvec2, PavFOM2,  s = 'FOM_b2p')
simu2spe.plot_radiation_factor(ax, freqvec2spe, PavFOM2spe,  s = 'FOM_b2pspe')
simu3.plot_radiation_factor(ax, freqvec3, PavFOM3,  s = 'FOM_b3p')
if comsol_data:
    ax.plot(frequency, results, c = 'black', label=r'$\sigma_{COMSOL}$')
    ax.legend()

plot_analytical_result = True
if plot_analytical_result:
    plot_analytical_result_sigma(ax, freqvec, radius)
plt.savefig("test.png")


from operators_POO import least_square_err, compute_analytical_radiation_factor

Z_ana = compute_analytical_radiation_factor(freqvec, radius)

err_B1p = least_square_err(freqvec, Z_ana.real, freqvec1, simu1.compute_radiation_factor(freqvec1, PavFOM1).real)
print(f'For lc = {lc} - L2_err(B1p) = {err_B1p}')

err_B2p = least_square_err(freqvec, Z_ana.real, freqvec2, simu2.compute_radiation_factor(freqvec2, PavFOM2).real)
print(f'For lc = {lc} - L2_err(B2p) = {err_B2p}')

err_B2p_tang = least_square_err(freqvec, Z_ana.real, freqvec2spe, simu2spe.compute_radiation_factor(freqvec2spe, PavFOM2spe).real)
print(f'For lc = {lc} - L2_err(err_B2p_tang) = {err_B2p_tang}')

err_B3p = least_square_err(freqvec, Z_ana.real, freqvec3, simu3.compute_radiation_factor(freqvec3, PavFOM3).real)
print(f'For lc = {lc} - L2_err(err_B3p) = {err_B3p}')
