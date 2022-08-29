import sys
from abipy.ppcodes.oncv_parser import OncvParser
import matplotlib.pyplot as plt

def _color_set(l):
    # color set
    if l == 0:
        color = 'blue'
    elif l == 1:
        color = 'red'
    elif l == 2:
        color = 'black'
    else:
        color = 'green'
        
    return color

def run():
    
    fig, axs = plt.subplots(2, 1)
    
    pspout = sys.argv[1]
    p = OncvParser(pspout)
    p.scan(verbose=1)
    
    logders = p.atan_logders
    
    for ld in logders.ae.values():
        # line color set
        color = _color_set(l=ld.l)
            
        axs[0].plot(ld.energies, ld.values, label=f"AE: l={ld.l}", color=color, linestyle='dashed')
        
    for ld in logders.ps.values():
        # line color set
        color = _color_set(l=ld.l)
        
        axs[0].plot(ld.energies, ld.values, label=f"PS: l={ld.l}", color=color, linestyle='solid')
    
    axs[0].set_xlabel("energies (Ha)")
    axs[0].set_ylabel("Atan Logders")
    axs[0].legend()
    
    k_ecut = p.kene_vs_ecut
    ecut_hint_low, ecut_hint_high = p.hints["low"]["ecut"], p.hints["high"]["ecut"]
    
    for convdata in k_ecut.values():
        color = _color_set(l=convdata.l)
        
        axs[1].plot(convdata.energies, convdata.values, label=f"l={convdata.l}", color=color)
        
    # plot ecut hint
    axs[1].axvline(x=ecut_hint_low, label="low hint ecut", linestyle="dotted")
    axs[1].axvline(x=ecut_hint_high, label="high hint ecut", linestyle="dotted")
        
    axs[1].set_xlabel("energies (Ha)")
    axs[1].set_ylabel("Kinetic energy")
    axs[1].legend()
    
    fig.tight_layout()
    plt.savefig('temp.png')

if __name__ == '__main__':
    run()