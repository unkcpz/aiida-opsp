"""
The script will go through the folder contain upf (by ONCV) files,
parse the input parameters of ONCV and 
plot the range of every parameters. The x axis is the elements, y is the value of 
ONCV input parameters.

rc:
qc:
rc/qc: 
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    psp_folder = sys.argv[1]
    dir = os.path.abspath(psp_folder)
    lst_element = []
    lst_z = []
    # lst_rc = []
    # lst_qcut = []
    lst_rrcqcut = {
        '0': [],
        '1': [],
        '2': [],
        '3': [],
    }
    lst_rc = {
        '0': [],
        '1': [],
        '2': [],
        '3': [],
    }
    lst_qcut = {
        '0': [],
        '1': [],
        '2': [],
        '3': [],
    }
    lst_debl = {
        '0': [],
        '1': [],
        '2': [],
        '3': [],
    }
    lst_rc5 = []
    for filename in os.listdir(dir):
        inpl = []
        if filename.endswith('.upf') or filename.endswith('.UPF'):
            f = os.path.join(dir, filename)
            with open(f, 'r') as fh:
                lines = fh.readlines()
                
                for i, line in enumerate(lines):
                    if '<PP_INPUTFILE>' in line:
                        start_idx = i
                        
                    if '</PP_INPUTFILE>' in line:
                        end_idx = i
  
                inp_lines = lines[start_idx+1:end_idx]

            l = []
            rc = []
            qcut = []
            debl = []
            for i, line in enumerate(inp_lines):
                if '# atsym' in line:
                    s = inp_lines[i+1]
                    element, z, nc, nv, iexc, psfile = s.split()
                    z = float(z)
                    nc = int(nc)
                    nv = int(nv)
                    continue
                
                if '# lmax' in line:
                    s = inp_lines[i+1]
                    lmax = s.strip()
                    lmax = int(lmax)
                    continue
                
                if 'nbas,' in line:
                    for ii in range(lmax+1):
                        s = inp_lines[i+1 + ii]
                        tmp_l, tmp_rc, _, _, _, tmp_qcut = s.split()
                        l.append(int(tmp_l))
                        rc.append(float(tmp_rc))
                        qcut.append(float(tmp_qcut))
                        
                    continue
                
                if 'nproj,' in line:
                    for ii in range(lmax+1):
                        s = inp_lines[i+1 + ii]
                        _, _, tmp_debl = s.split()
                        debl.append(float(tmp_debl))
                        
                    continue
                
                if 'rc(5),' in line:
                    s = inp_lines[i+1]
                    lloc, lpopt, rc5, dvloc0 = s.split()
                    lloc = int(lloc)
                    lpopt = int(lpopt)
                    rc5 = float(rc5)
                    dvloc0 = float(dvloc0)
                    
                    continue
                        
        lst_element.append(element)
        lst_z.append(z)
        lst_rc5.append(rc5)

        for i in [0, 1, 2, 3]:
            try:
                lst_rrcqcut[f'{i}'].append(rc[i]*qcut[i]/10)
            except Exception:
                lst_rrcqcut[f'{i}'].append(None)
                
            try:
                lst_rc[f'{i}'].append(rc[i])
            except Exception:
                lst_rc[f'{i}'].append(None)
                
            try:
                lst_qcut[f'{i}'].append(qcut[i])
            except Exception:
                lst_qcut[f'{i}'].append(None)
                
            try:
                lst_debl[f'{i}'].append(debl[i])
            except Exception:
                lst_debl[f'{i}'].append(None)

          
    # Plot  
    fig, axs = plt.subplots(4, 1, figsize=(20,6*4))
    axs[0].scatter(lst_z, np.array(lst_rrcqcut['0']), marker='o', label='l=0')
    axs[0].scatter(lst_z, np.array(lst_rrcqcut['1']), marker='^', label='l=1')
    axs[0].scatter(lst_z, np.array(lst_rrcqcut['2']), marker='s', label='l=2')
    axs[0].scatter(lst_z, np.array(lst_rrcqcut['3']), marker='P', label='l=3')
    axs[0].set_xticks(lst_z, lst_element)
    axs[0].set_ylim((0.5, 3.0))
    axs[0].set_ylabel('rrkj: rc*qcut/10')
    axs[0].legend()
    
    axs[1].scatter(lst_z, np.array(lst_rc['0']), marker='o', label='l=0')
    axs[1].scatter(lst_z, np.array(lst_rc['1']), marker='^', label='l=1')
    axs[1].scatter(lst_z, np.array(lst_rc['2']), marker='s', label='l=2')
    axs[1].scatter(lst_z, np.array(lst_rc['3']), marker='P', label='l=3')
    # also rc(5) here for comparism
    axs[1].scatter(lst_z, np.array(lst_rc5), marker='p', label='local potential rc')
    axs[1].set_xticks(lst_z, lst_element)
    axs[1].set_ylim((0.5, 4.0))
    axs[1].set_ylabel('rrkj: rc')
    axs[1].legend()
    
    axs[2].scatter(lst_z, np.array(lst_qcut['0']), marker='o', label='l=0')
    axs[2].scatter(lst_z, np.array(lst_qcut['1']), marker='^', label='l=1')
    axs[2].scatter(lst_z, np.array(lst_qcut['2']), marker='s', label='l=2')
    axs[2].scatter(lst_z, np.array(lst_qcut['3']), marker='P', label='l=3')
    axs[2].set_xticks(lst_z, lst_element)
    axs[2].set_ylim((4.0, 11.0))
    axs[2].set_ylabel('rrkj: qcut')
    axs[2].legend()
    
    axs[3].scatter(lst_z, np.array(lst_debl['0']), marker='o', label='l=0')
    axs[3].scatter(lst_z, np.array(lst_debl['1']), marker='^', label='l=1')
    axs[3].scatter(lst_z, np.array(lst_debl['2']), marker='s', label='l=2')
    axs[3].scatter(lst_z, np.array(lst_debl['3']), marker='P', label='l=3')
    axs[3].set_xticks(lst_z, lst_element)
    axs[3].set_ylim((0.5, 5.0))
    axs[3].set_ylabel('ov: debl')
    axs[3].legend()
    
    plt.savefig(f'{psp_folder}.png')