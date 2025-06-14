import numpy as np
import pandas as pd

import LST as element
import solvers as solvers
import postLST as post

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as tri
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap


# 1) Define the three key colours of the Mexican flag
mexican_colors = [
    "#006847",  # green
    "#FFFFFF",  # white
    "#CE1126",  # red
]

# 2) Build a smooth colormap that transitions from green → white → red
mexican_cmap = LinearSegmentedColormap.from_list(
    "mexican_flag",   # name (only used if you choose to register later)
    mexican_colors,
    N=256             # number of discrete levels in the map
)


def ITC_S0_E1(X,T,G,Df,V0,V1,t):
    nel = len(T) # number of elements

    # Pre-init - Saving stresses/strains at each node in array, (nd, xx, yy, xy)
    MaxBet_el = np.zeros([nel*6,2])

    for el in range(1,nel+1):
        eli = el-1 # element index
        #nodes
        no1, no2, no3 = T[eli,0], T[eli,1], T[eli,2] # first, second, third
        no4, no5, no6 = T[eli,3], T[eli,4], T[eli,5] # fourth, fith, sixth
        nodes_T = np.array([no1, no2, no3, no4, no5, no6]) # nodes loaded into array to enumerate stresses
        # index
        no1i = no1-1 # node 1
        no2i = no2-1 # node 2
        no3i = no3-1 # node 3
        # coordinates
        X1 = X[no1i]  # coordinates 1
        X2 = X[no2i]  # coordinates 2
        X3 = X[no3i]  # coordinates 3
        Ge = G[eli] # Material properties
        de = Df[eli,:] # index array for dofs
        dei = de-1 # dof index
        Ve0 = V0[dei] # displacement vector
        Ve1 = V1[dei] # displacement vector

        for i,z in enumerate([[1,0,0], [0,1,0], [0,0,1], [0,1/2,1/2], [1/2,0,1/2], [1/2,1/2,0]]): #[0,1/2,1/2], [1/2,0,1/2], [1/2,1/2,0]
            SS_el, Ee_el = element.S_Eps(X1,X2,X3,Ge,Ve0,Ve1,z) #Generalised stresses: xx, yy, xy
            # Keeping track of node numbers
            MaxBet_el[eli*6+i, 0] = np.int32(el)
            # Saving calculated IfR-values
            MaxBet_el[eli*6+i,1] = -np.dot(SS_el,Ee_el) / t

    return MaxBet_el

    
def plotITC_4x(X, T, nel, MaxBet_el, fac,use_sci):
    '''
    Plot generalized stress change (IfR) based on Maxwell-Betti.
        X: Coordinate matrix
        T: Topology matrix
        nel: number of elements
        MaxBet_el: array of element-wise IfR values
        fac: scaling factor for color normalization
    '''
    fig, ax = plt.subplots(figsize=[9, 6])
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    lim = 1.0
    ax.set_xlim(np.floor(np.min(X[:, 0])) - 0.5, np.ceil(np.max(X[:, 0])) + 0.5)
    ax.set_ylim(np.floor(np.min(X[:, 1])) - 0.5, np.ceil(np.max(X[:, 1])) + 0.5)

    x_ticks = np.arange(np.floor(np.min(X[:, 0])) - 0, np.ceil(np.max(X[:, 0]))+0.1 , lim)
    y_ticks = np.arange(np.floor(np.min(X[:, 1])) - 0, np.ceil(np.max(X[:, 1]))+0.1 , lim)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    vminp = min(np.min(MaxBet_el), 0) * fac
    vmaxp = max(np.max(MaxBet_el), 0) * fac
    norm = mpl.colors.Normalize(vmin=vminp, vmax=vmaxp)

    for el in range(1, nel + 1):
        eli = el - 1
        no1, no2, no3 = T[eli, 0], T[eli, 1], T[eli, 2]
        no4, no5, no6 = T[eli, 3], T[eli, 4], T[eli, 5]
        no1i, no2i, no3i = no1 - 1, no2 - 1, no3 - 1
        no4i, no5i, no6i = no4 - 1, no5 - 1, no6 - 1
        nodesi = [no1i, no2i, no3i, no4i, no5i, no6i]

        IfR_el = MaxBet_el[eli * 6: eli * 6 + 6, 1]

        coordinates = np.array([X[no1i], X[no2i], X[no3i], X[no4i], X[no5i], X[no6i]])

        for i in [[0, 5, 4], [5, 1, 3], [4, 3, 2], [5, 3, 4]]:
            values = np.array([IfR_el[i[0]], IfR_el[i[1]], IfR_el[i[2]]])
            points = np.array([coordinates[i[0]], coordinates[i[1]], coordinates[i[2]]])
            triang = tri.Triangulation(points[:, 0], points[:, 1], [[0, 1, 2]])
            ax.tripcolor(triang, values, shading='gouraud', cmap='turbo', norm=norm)


        # Optional: element outline
        x_coords = [X[no1i, 0], X[no6i, 0], X[no2i, 0], X[no4i, 0],
                    X[no3i, 0], X[no5i, 0], X[no1i, 0]]
        y_coords = [X[no1i, 1], X[no6i, 1], X[no2i, 1], X[no4i, 1],
                    X[no3i, 1], X[no5i, 1], X[no1i, 1]]
        ax.plot(x_coords, y_coords, color='black', linewidth=0.4, alpha=0.34, linestyle='-')

    ax.set_title("Influence of Thickness Change")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)

    cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap='turbo'),
            cax=cax, orientation='vertical')

    cbar.set_label('IfR', rotation=270, labelpad=15)

    # build tick array and drop any tick that is (numerically) 0
    ticks = np.linspace(vminp, vmaxp, 10)
    ticks = [t for t in ticks if not np.isclose(t, 0.0)]
    cbar.set_ticks(ticks)

    # use scientific notation or plain numbers (switch on/off here)            # <-- toggle
    if use_sci:
        cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))
    else:
        cbar.ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        cbar.ax.ticklabel_format(useOffset=False, style='plain')

    # keep ticks on the right only
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')
    cax.tick_params(axis='y', which='both', right=True, left=False)




