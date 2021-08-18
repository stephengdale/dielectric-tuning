# A minimal script for finding the E_N, E_{N-1}, \epsilon_N and \epsilon_N-1 for a given neutral singlet molecule when using and/or tuning the \alpha value of B\alphaLYP
# Commented lines include optional settings to aid SCF convergence, in some cases they will need to be moved into a set_options command.
# This script was used on the VIE dataset of McKechie et al. (S. McKechnie, G. H. Booth, A. J. Cohen,   and J. M. Cole, “On the accu-racy of density functional theory and wave function methods for calculatingvertical ionization energies,” J. Chem. Phys.142, 194114 (2015).)
# xyz coordinates were taken from 20_mbcc_vie of https://github.com/aoterodelaroza/refdata.

import psi4
import numpy as np
import scipy.integrate as integrate

from  optparse import OptionParser

def genAll(FileName="CH3"):
    print(FileName)

    cat = FileName + "+"
    neu = FileName
    
    
    xyz_cat = "".join(open("/home/stephen/git/refdata/20_mbcc_vie/" + cat + ".xyz", 'r').readlines()[1:]) + 'symmetry c1'
    xyz_neu = "".join(open("/home/stephen/git/refdata/20_mbcc_vie/" + neu + ".xyz", 'r').readlines()[1:]) + 'symmetry c1'
    psi4.set_output_file(neu + ".out")
    psi4.set_options({  'reference' : 'uks'})
    eV = 27.2114 
    
    def diff(x, a_ene, a_homo, n_ene, n_lumo):
        dE = a_ene - n_ene
        q = dE * x + x * (1-x) * ((n_lumo - dE) * (1-x) + (-a_homo + dE) * x)
        q = q*eV
        return q
    
    def diffi(x, a_ene, a_homo, n_ene, n_lumo):
        dE = a_ene - n_ene
        q = x * (1-x) * ((n_lumo - dE) * (1-x) + (-a_homo + dE) * x)
        q = q*eV
        return q
    
    
    
    def integral_at_global_hybrid(alpha):
    
        psi4.set_options({  'reference' : 'uks'})
        psi4.set_options({  'reference' : 'uks',
                            'basis' : 'aug-cc-pvdz'})
                            # 'soscf' : True,
							# 'maxiter' : 200,
        
        custom = {"name" : "global_hybrid", 
                  "x_functionals" : {"GGA_X_B88" : {"alpha" : 1.0-alpha}}, 
                  "x_hf" : {"alpha" : 0.0+alpha},
                  "c_functionals" : {"GGA_C_LYP" : {}}
                  }
        
        psi4.geometry(xyz_cat)
        psi4.set_options({  'guess' : 'sad'})
        # ene_a, wfn_a = psi4.energy('SCF', dft_functional='b3lyp', return_wfn=True)
        # psi4.set_options({  'guess' : 'read'})
        ene_a, wfn_a = psi4.energy('SCF', dft_functional=custom, return_wfn=True)
        LUMO = wfn_a.epsilon_b_subset("AO","ALL").nph[0][wfn_a.nalpha()-1]
        psi4.core.clean()
        
        psi4.geometry(xyz_neu)
        psi4.set_options({  'guess' : 'sad'})
        # ene_b, wfn_b = psi4.energy('scf', dft_functional='b3lyp', return_wfn=True)
        # psi4.set_options({  'guess' : 'read'})
        ene_b, wfn_b = psi4.energy('scf', dft_functional=custom, return_wfn=True)
        HOMO = wfn_b.epsilon_a_subset("AO","ALL").nph[0][wfn_b.nalpha()-1]
        psi4.core.clean()
        
        return (integrate.quad(lambda x: diffi(x, ene_b, HOMO, ene_a, LUMO),0,1)[0], ene_b, HOMO, ene_a, LUMO)
    
    def decide(integrals, alphas, j):
        log = open(neu + ".log", 'a')
        log.write("{:20s} |  {:20.10f} {:20.10f} |  {:20.10f} {:20.10f} {:3}\n".format(neu, integrals[0], integrals[1], alphas[0], alphas[1], j))
        j += 1
        if j > 15:
            return alphas[list(map(abs, integrals)).index(min(list(map(abs, integrals))))]
        elif integrals[1] < 0:
            return alphas[1]
        elif integrals[0] > 0:
            return alphas[0]
        else:
            tmp = integral_at_global_hybrid(sum(alphas)/2)[0]
            if tmp > 0:
                integrals[1]   = tmp
                alphas[1] = sum(alphas)/2
            else:
                integrals[0]   = tmp
                alphas[0] = sum(alphas)/2
            return decide(integrals, alphas, j)
    
    keep = open(neu + ".dat","w")
    dat = integral_at_global_hybrid(0.0)
    keep.write('{:10.4f} : {:10.4f} {:10.4f} {:10.4f} {:10.4f} \n'.format(0.0, dat[1], dat[2], dat[3], dat[4]))
    print('{:10.4f} : {:10.4f} {:10.4f} {:10.4f} {:10.4f}'.format(0.0, dat[1], dat[2], dat[3], dat[4]))
   
    # BHHLYP
    dat = integral_at_global_hybrid(0.5)
    keep.write('{:10.4f} : {:10.4f} {:10.4f} {:10.4f} {:10.4f} \n'.format(0.5, dat[1], dat[2], dat[3], dat[4]))
    print('{:10.4f} : {:10.4f} {:10.4f} {:10.4f} {:10.4f}'.format(0.5, dat[1], dat[2], dat[3], dat[4]))
    
    # Full HF
    dat = integral_at_global_hybrid(1.0)
    keep.write('{:10.4f} : {:10.4f} {:10.4f} {:10.4f} {:10.4f} \n'.format(1.0, dat[1], dat[2], dat[3], dat[4]))
    print('{:10.4f} : {:10.4f} {:10.4f} {:10.4f} {:10.4f}'.format(1.0, dat[1], dat[2], dat[3], dat[4]))
    
    # Optimized Calculations and Plotting
    alphas = [0.0, 1.0]
    integrals   = [0 ,0]
    
    integrals[0] = integral_at_global_hybrid(alphas[0])[0]
    integrals[1] = integral_at_global_hybrid(alphas[1])[0]
    
    opt = decide(integrals, alphas, 0)

    dat = integral_at_global_hybrid(opt)
    keep.write('{:10.4f} : {:10.4f} {:10.4f} {:10.4f} {:10.4f} \n'.format(opt, dat[1], dat[2], dat[3], dat[4]))
    print('{:10.4f} : {:10.4f} {:10.4f} {:10.4f} {:10.4f}'.format(opt, dat[1], dat[2], dat[3], dat[4]))
    keep.close()
    
    return None

genParse = OptionParser()
genParse.add_option('-M', type="string", default="NH3")
(options, args) = genParse.parse_args()

genAll(options.M)




