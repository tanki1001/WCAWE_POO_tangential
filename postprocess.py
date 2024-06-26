import numpy as np
import matplotlib.pyplot as plt
from scipy import special


rho0 = 1.21
c0 = 343.8

def relative_errZ(freqvec, Z_analytical, Z_center1):

    relative_err_ReZ1 = [0 for i in range(len(freqvec))]
    relative_err_ImZ1 = [0 for i in range(len(freqvec))]
    for ii in range(len(freqvec)):
        relative_err_ReZ1[ii] = abs(Z_center1.real[ii]-Z_analytical.real[ii])/Z_analytical.real[ii]
        relative_err_ImZ1[ii] = abs(Z_center1.imag[ii]-Z_analytical.imag[ii])/Z_analytical.imag[ii]
    ev_relative_err_ReZ1 = sum(relative_err_ReZ1)/len(relative_err_ReZ1)
    ev_relative_err_ImZ1 = sum(relative_err_ImZ1)/len(relative_err_ImZ1)
    
    return ev_relative_err_ReZ1, ev_relative_err_ImZ1


def import_FOM_result(s):
    
    with open(s+".txt", "r") as f:
        frequency = list()
        results = list()
        for line in f:
            if "%" in line:
                continue
            data = line.split()
            frequency.append(data[0])
            results.append(data[2])
            frequency = [float(element) for element in frequency]
            results = [float(element) for element in results]
    return frequency, results



