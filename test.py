import numpy as np
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

freqvec1 = np.arange(80,2001,20)
freqvec2 = np.arange(1250, 1501, 20)
print(set(freqvec2) <= set(freqvec1))
