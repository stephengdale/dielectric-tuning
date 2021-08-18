# A minimal script for finding the E_N, E_{N-1}, \epsilon_N and \epsilon_N-1 for a given neutral singlet molecule when using a CPCM solvent model and/or tuning the dielectric constant to achieve linear behaviour.
# Commented lines include optional settings to aid SCF convergence, in some cases they will need to be moved into a set_options command.
# This script was used on the VIE dataset of McKechie et al. (S. McKechnie, G. H. Booth, A. J. Cohen,   and J. M. Cole, “On the accu-racy of density functional theory and wave function methods for calculatingvertical ionization energies,” J. Chem. Phys.142, 194114 (2015).)
# xyz coordinates were taken from 20_mbcc_vie of https://github.com/aoterodelaroza/refdata.

import psi4
import numpy as np
import scipy.integrate as integrate
import subprocess
import os

from  optparse import OptionParser

def genAll(FileName="CH3"):
    print(FileName)

    cat = FileName + "+"
    neu = FileName
    
    xyz_cat = "".join(open("/home/stephen/git/refdata/20_mbcc_vie/" + cat + ".xyz", 'r').readlines()[1:]) + 'symmetry c1'
    xyz_neu = "".join(open("/home/stephen/git/refdata/20_mbcc_vie/" + neu + ".xyz", 'r').readlines()[1:]) + 'symmetry c1'
    psi4.set_output_file(neu + ".out")
    psi4.set_options({  'reference' : 'uks'})
    X = np.arange(0.,1.01,0.01)
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
    
    
    
    def integral_at_dielectric(func, diele):
    
        psi4.set_options({  'reference' : 'uks'})
        psi4.set_options({  'reference' : 'uks',
                            'basis' : 'aug-cc-pvdz',
                            'pcm' : True,
                            'pcm_scf_type' : 'total'})
                            # 'maxiter' : 200,
                            # 'soscf' : True,
        
        start = '''
            Units = Angstrom
            Medium {
            SolverType = cpcm
            Solvent = Explicit
            ProbeRadius=2.0
            Green<inside> {Type=Vacuum}
            Green<outside> {Type=UniformDielectric 
                            Eps='''
        stop = '''                    EpsDyn=2.0}
            }
            Cavity {
            RadiiSet = UFF
            Type = GePol
            Scaling = False
            Area = 0.3
            Mode = Implicit
            }
        '''
        pcm_string = start+str(diele)+'\n'+stop
    
        psi4.pcm_helper(pcm_string)
    
        psi4.geometry(xyz_cat)
        psi4.set_options({  'guess' : 'sad'})
        # psi4.set_options({  'damping_percentage' : 0})
        # ene_a, wfn_a = psi4.energy('b3lyp', return_wfn=True)
        # psi4.set_options({  'guess' : 'read'})
        # psi4.set_options({  'damping_percentage' : 50})
        ene_a, wfn_a = psi4.energy(func, return_wfn=True)
        LUMO = wfn_a.epsilon_b_subset("AO","ALL").nph[0][wfn_a.nalpha()-1]
        psi4.core.clean()
        
        psi4.geometry(xyz_neu)
        psi4.set_options({  'guess' : 'sad'})
        # psi4.set_options({  'damping_percentage' : 0})
        # ene_b, wfn_b = psi4.energy('b3lyp', return_wfn=True)
        # psi4.set_options({  'guess' : 'read'})
        # psi4.set_options({  'damping_percentage' : 50})
        ene_b, wfn_b = psi4.energy(func, return_wfn=True)
        HOMO = wfn_b.epsilon_a_subset("AO","ALL").nph[0][wfn_b.nalpha()-1]
        psi4.core.clean()
        
        return (integrate.quad(lambda x: diffi(x, ene_b, HOMO, ene_a, LUMO),0,1)[0], ene_b, HOMO, ene_a, LUMO)
    
    def decide(func, integrals, dielectrics, j):
        log = open(neu + ".log", 'a')
        log.write("{:20s} {:10s} |  {:20.10f} {:20.10f} |  {:20.10f} {:20.10f} {:3}\n".format(neu, func, integrals[0], integrals[1], dielectrics[0], dielectrics[1], j))
        j += 1
        if j > 15:
            return dielectrics[list(map(abs, integrals)).index(min(list(map(abs, integrals))))]
        elif integrals[1] < 0:
            return dielectrics[1]
        elif integrals[0] > 0:
            return dielectrics[0]
        else:
            tmp = integral_at_dielectric(func, sum(dielectrics)/2)[0]
            if tmp > 0:
                integrals[1]   = tmp
                dielectrics[1] = sum(dielectrics)/2
            else:
                integrals[0]   = tmp
                dielectrics[0] = sum(dielectrics)/2
            return decide(func, integrals, dielectrics, j)
    
    occ = np.arange(0., 1.1, 0.1) 
    funcs = ['blyp', 'b3lyp', 'bhhlyp', 'hf', 'hse06', 'lrc-wpbe'] # 'blyp', 'b3lyp', 'bhhlyp', 'hf', 
    cols  = ['b'   ,     'g',      'r',  'k',     'o',        'p'] # 'b'   ,     'g',      'r',  'k', 
    
    # Bare DFT
    if not os.path.isfile('{}_bare.dat'.format(neu)):
        keep = open(neu + "_bare.dat","w")
        keep.close()
    if not str(subprocess.run('wc -l {}_bare.dat'.format(neu), check=True, shell=True, capture_output=True).stdout).split()[0][-1] == '6':
        keep = open(neu + "_bare.dat","w")
        i = 0
        while i < len(funcs):
            dat = integral_at_dielectric(funcs[i], 1.01)
            keep.write('{:10s} : {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} \n'.format(funcs[i], 1.01, dat[1], dat[2], dat[3], dat[4]))
            print('{:10s} : {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f}'.format(funcs[i], 1.01, dat[1], dat[2], dat[3], dat[4]))
            i += 1
        keep.close()
        
        # DFT with water pcm
        
    if not os.path.isfile('{}_full.dat'.format(neu)):
        keep = open(neu + "_full.dat","w")
        keep.close()
    if not str(subprocess.run('wc -l {}_full.dat'.format(neu), check=True, shell=True, capture_output=True).stdout).split()[0][-1] == '6':
        keep = open(neu + "_full.dat","w")
        i = 0
        while i < len(funcs):
            dat = integral_at_dielectric(funcs[i], 80.0)
            keep.write('{:10s} : {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} \n'.format(funcs[i], 80.0, dat[1], dat[2], dat[3], dat[4]))
            print('{:10s} : {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f}'.format(funcs[i], 80.0, dat[1], dat[2], dat[3], dat[4]))
            i += 1
        keep.close()
    
    # Optimized Calculations and Plotting
    
    if not os.path.isfile('{}_opt.dat'.format(neu)):
        keep = open(neu + "_opt.dat","w")
        keep.close()
    if not str(subprocess.run('wc -l {}_opt.dat'.format(neu), check=True, shell=True, capture_output=True).stdout).split()[0][-1] == '6':
        keep = open(neu + "_opt.dat","w")
        i = 0
        while i < len(funcs):
            dielectrics = [1.01, 80.0]
            integrals   = [0 ,0]
        
            integrals[0] = integral_at_dielectric(funcs[i], dielectrics[0])[0]
            integrals[1] = integral_at_dielectric(funcs[i], dielectrics[1])[0]
            
            opt = decide(funcs[i], integrals, dielectrics, 0)
            # bare = integral_at_dielectric(funcs[i], 1.01)
            # full = integral_at_dielectric(funcs[i], 80.0)
            dat = integral_at_dielectric(funcs[i], opt)
            keep.write('{:10s} : {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} \n'.format(funcs[i], opt, dat[1], dat[2], dat[3], dat[4]))
            print('{:10s} : {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f}'.format(funcs[i], opt, dat[1], dat[2], dat[3], dat[4]))
            i += 1
        keep.close()
        
    return None

genParse = OptionParser()
genParse.add_option('-M', type="string", default="NH3")
(options, args) = genParse.parse_args()

genAll(options.M)




