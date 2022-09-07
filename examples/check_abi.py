import os
import sys
from abipy.ppcodes.oncv_parser import OncvParser
import matplotlib.pyplot as plt
from aiida import orm
import aiida

aiida.load_profile("q0")

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
    
    pk = sys.argv[1]
    try:
        n = orm.load_node(pk)
        pspout = os.path.join(n.outputs.remote_folder.get_remote_path(), 'aiida.out')
    except:
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
    
    axs[0].axvline(x=-2, linestyle="dotted")
    axs[0].axvline(x=0, linestyle="dotted")
    axs[0].axvline(x=2, linestyle="dotted")
    axs[0].axvline(x=5, linestyle="dotted")
    axs[0].set_xlabel("energies (Ha)")
    axs[0].set_ylabel("Atan Logders")
    axs[0].legend(loc="upper right", prop={'size': 7})
    
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
    try:
        filename = sys.argv[2]
    except:
        filename = "temp.png"
    plt.savefig(filename)

if __name__ == '__main__':
    run()