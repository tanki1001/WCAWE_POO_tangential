%%%Author : Pierre MARIOTTI
%%%Location : KTH - Royal Institute of Technology
%%%Last edit : 19/09/23

--------------------------
Description
--------------------------

This folder provides an implementation of a model order reduction method, the well conditionned asymptotic waveform equation (WCAWE). Various FE problem can be added, but the implementation is, so far, related to the study of a baffle case and the implementation of an absorbing boundary condition (ABC). Look into the README.txt file in the 'Baffle_sudy' folder to learn more about it. It only, so far, implements the 1st order of the BGT operator, similarly, other operators and operators order can be added.
The simulation is performed using the open source finite element sorfware FEniCSx.

--------------------------
Programs
--------------------------

In this repertory, one can find a main.py which firstly import all the needed modules from geometries.py, operators.py which are the important files.
Within the main.py file, one can copy past as many frequency sweep as it wants to compare them. 
Withing the operators.py file, one might find all the needed functions. The 2 most import ones are, z_f_matrices_b1p_withDiff(mesh_info, submesh_info, freq0, nb_d) to obtain the FE matrix Z and the force vector F at the frequency freq0 from the mesh and the submesh. The parameter nb_d is for the number of derivates of the matrices need, 0 is for the use without any derivates (when it is needed to reconstruct the matrix at each frequency).  basis_N_WCAWE(list_Z, list_F, freq0, N) to contruct the basis of N vectors for a list of FE matrices and force vector. 

--------------------------
Practical use
--------------------------
The idea is, if a function such as z_f_matrices_b1p_withDiff is built for any problemes (other BGT degrees for instance), the list of Z and F matrices can be used on the function basis_N_WCAWE. If the user knows what is the order of the developped matricial polynom it is stable. Other informations are provided in the comments if the code. 

So far, 2 frequency sweeps are performed. The first one is the method performed one time at a frequency in the middle of the frenquency range. The second one is the method applied in 2 different frequencies. The number of vectors for the basis can be changed, even between the 2 frequency sweeps of the same study.
Pictures and text files are written in order to be reused. They are titled to provide information on each case has been perfomed, at wich frequency, either one or two times.

--------------------------
TODO
--------------------------
Compared to the baffle_study folder, only the cubic geometrie has been implemented, only the first order of the BGT operator has been implemented, but that is not the main issue :
The method can be really efficient, but still provides instable results at high frequencies. But also when the number of vector is too high.
For instance, about the case where the WCAWE is used two times, the first frequency range [80-1040]Hz can be almost perfectly fitted with only 10 vectors whereas the second frequency range [1040 - 2000] Hz can't be fitted at all, 60 is the best, but more provides numerical instabilities. 

