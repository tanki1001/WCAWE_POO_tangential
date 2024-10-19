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

from operators_POO import (Mesh, Loading, Simulation,
                        B1p, B2p, B3p,
                        B2p_modified_r, B3p_modified_r,
                        SVD_ortho,
                        store_results, store_results_wcawe, import_frequency_sweep, import_frequency_sweepv2,
                        least_square_err, compute_analytical_radiation_factor,
                        get_wcawe_param, parse_wcawe_param)

file_wcawe_para = True
if file_wcawe_para:
    dir, geo, case, ope, lc, dimP, dimQ = get_wcawe_param()
    geometry1 = geo
    geometry2 = case
else:
    geometry1 = 'cubic'
    geometry2 = 'large'

geometry  = geometry1 + '_'+ geometry2

if   geometry2 == 'small':
    side_box = 0.11
    lc       = 8e-3
elif geometry2 == 'large':
    side_box = 0.40
    lc        = 2e-2
else :
    print("Enter your own side_box and mesh size in the code")
    side_box = 0.11
    lc       = 8e-3

freqvec = np.arange(80, 2001, 20)
radius   = 0.1
rho0 = 1.21
c0   = 343.8

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

def fct_main_wcawe(
    degP,
    degQ,
    str_ope,
    freqvec,
    list_N,
    list_freq,
    file_name,
    save_fig,
    save_data,
):
    file_wcawe_para_list = [file_wcawe_para, list_freq, list_N]

    dimP = degP
    if False :
        dimQ = deg -1
    else :
        dimQ = degP
    mesh_    = Mesh(dimP, dimQ, side_box, radius, lc, geo_fct)

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

    loading = Loading(mesh_)
    simu    = Simulation(mesh_, ope, loading)
    if "modified" in str_ope:
        s = geometry + '_' + str_ope + '_' + str(lc) + '_' + str(dimP) + '_' + str(dimQ)
    else :
        s = 'tangential_'+ geometry + '_' + str_ope + '_' + str(lc) + '_' + str(dimP) + '_' + str(dimQ)
    #freqvec_fct, PavFOM_fct = import_frequency_sweepv2(s)
    freqvec_fct = np.arange(80, 2001, 20) 
    

    t1   = time()
    Vn   = simu.merged_WCAWE(list_N, list_freq)
    t2   = time()
    print(f'WCAWE CPU time  : {t2 -t1}')

    Vn = SVD_ortho(Vn)
    t3 = time()
    print(f'SVD CPU time  : {t3 -t2}')
    PavWCAWE_fct = simu.moment_matching_MOR(Vn, freqvec_fct)
    t4 = time()
    print(f'Whole CPU time  : {t4 -t1}')

    #err_wcawe = least_square_err(freqvec_fct, PavFOM_fct.real, freqvec_fct, simu.compute_radiation_factor(freqvec_fct, PavWCAWE_fct).real)
    #print(f'For list_N = {list_N} - L2_err(wcawe) = {err_wcawe}')

    if save_fig:
        fig, ax = plt.subplots()
        simu.plot_radiation_factor(ax, freqvec_fct, PavFOM_fct, s = 'FOM_' + str_ope, compute = False)
        simu.plot_radiation_factor(ax, freqvec_fct, PavWCAWE_fct, s = 'WCAWE')
        ax.set_ylim(0, 1.5)
        plt.title(f'list_N = {list_N}')
        plt.savefig("/root/WCAWE_POO_github/"+ file_name + ".png")
    
    if save_data :
        list_s = [geometry1, geometry2, str_ope, str(lc), str(dimP), str(dimQ)]
        store_results_wcawe(list_s, freqvec, PavWCAWE_fct, simu, file_wcawe_para_list)

if file_wcawe_para:
    str_ope = ope
    list_freq, list_N = parse_wcawe_param()
else:
    str_ope = "b2p_modified_r"
    dimP = 3
    list_freq = [1000]
    list_N = [20]

if file_wcawe_para:
    fct_main_wcawe(
        degP      = dimP,
        degQ      = dimP,
        str_ope   = str_ope,
        freqvec   = freqvec,
        list_N    = list_N,
        list_freq = list_freq,
        file_name = "WCAWE_test",
        save_fig  = False,
        save_data = True
    )
else:
    dimP = 2
    dimQ = 2
    str_ope = "b2p_modified_r"
    freqvec = np.arange(80, 2001, 20)
    list_N = [5]
    list_freq = [1000]

    fct_main_wcawe(
        degP       = dimP,
        degQ       = dimP,
        str_ope   = str_ope,
        freqvec   = freqvec,
        list_N    = list_N,
        list_freq = list_freq,
        file_name = "WCAWE_test",
        save_fig  = False,
        save_data = True
    )

    dimP = 3
    dimQ = 3
    str_ope = "b3p_modified_r"
    freqvec = np.arange(80, 2001, 20)
    list_N = [5]
    list_freq = [1000]

    fct_main_wcawe(
        degP       = dimP,
        degQ       = dimP,
        str_ope   = str_ope,
        freqvec   = freqvec,
        list_N    = list_N,
        list_freq = list_freq,
        file_name = "WCAWE_test",
        save_fig  = False,
        save_data = True
    )

