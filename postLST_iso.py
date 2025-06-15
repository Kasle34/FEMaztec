import matplotlib.pyplot as plt
import numpy as np
from LST_iso import shape_funcs_LST_iso  
import LST as element

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as tri
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plotTop_iso(X,T,nno,nel,U,plot_no:bool,plot_supp:bool,steps:int):
    '''
    Plot topology
        X: Coordinate matrix
        T: Topology matrix
        nno: number of nodes
        nel: number of elements
    '''
    
    plt.figure(figsize=[12,8]) # Start figure
    plt.axis('equal') # set axis equal
    plt.xlabel('x') # Label x-axis
    plt.ylabel('y') # Label y-axis

    # If plot_nodes is True
    if plot_no:
        for no in range(1,nno+1):
            noi = no-1 #  node index
            # plot node number
            plt.text(X[noi,0],X[noi,1],str(no),\
                    bbox={'boxstyle':'round','facecolor':'lightgray', \
                        'alpha':0.5,'pad':0.1},\
                    horizontalalignment='center',verticalalignment='center',fontsize=8) 
            
    for el in range(1,nel+1):
        eli = el-1 # element index
        #nodes
        no1, no2, no3 = T[eli,0], T[eli,1], T[eli,2] # first, second, third
        no4, no5, no6 = T[eli,3], T[eli,4], T[eli,5] # fourth, fifth, sixth
        # index
        no1i, no2i, no3i = no1-1, no2-1, no3-1 # node 1, node 2, node 3
        no4i, no5i, no6i = no4-1, no5-1, no6-1 # node 4, node 5, node 6
        
        Xe = np.array([X[no1i],
                       X[no2i],
                       X[no3i],
                       X[no4i],
                       X[no5i],
                       X[no6i]])

        # Plotting edges:
        edge_node_indices = [[0, 5, 1], [1, 3, 2], [2, 4, 0]]
        for i, idx in enumerate(edge_node_indices):
            coords = []
            for t in np.linspace(0, 1, steps):
                if i == 0:
                    z = [t, 1 - t, 0]
                elif i == 1:
                    z = [0, t, 1 - t]
                else:
                    z = [1 - t, 0, t]
                N = shape_funcs_LST_iso(z)
                x = N[idx] @ Xe[idx, 0]  # Xe: shape (6, 2)
                y = N[idx] @ Xe[idx, 1]
                coords.append([x, y])
            # plot the coordinates
            coords = np.array(coords)
            #plt.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=0.5)
            colors = ['b', 'b', 'b']  # side 1 → blue, side 2 → green, side 3 → red
            plt.plot(coords[:, 0], coords[:, 1], color=colors[i], linewidth=0.8)
        
        # plotting element number
        Xm = np.mean([X[no1i,0], X[no2i,0], X[no3i,0]]) # x-coord
        Ym = np.mean([X[no1i,1], X[no2i,1], X[no3i,1]]) # y-coord
        plt.text(Xm,Ym,str(el),\
                 horizontalalignment='center',verticalalignment='center',fontsize=6) 

    # Plotting supports
    if plot_supp:
        for row in U:
            dofi = int(row[0])-1
            node = dofi // 2
            dir = dofi%2

            x, y = X[node]

            
            delta_X = max(X[:,0]) - min(X[:,0])
            delta_Y = max(X[:,1]) - min(X[:,1])
            scale = min(delta_X,delta_Y) * 0.03

            if dir == 0:
                # x-support (horizontal triangle)
                tri_x = [x - scale , x- scale, x]
                tri_y = [y - scale/2, y + scale/2, y]
            else:
                # y-support (vertical triangle)
                tri_x = [x - scale/2, x + scale/2, x]
                tri_y = [y - scale, y - scale, y]

            plt.fill(tri_x, tri_y, edgecolor='black', facecolor='grey',alpha=0.75)  
    
    # displaying the title
    plt.title("Topology")


def plotDof_iso(X, T, Df, V, nel, dispFac, steps:int): 
    '''
    Plot displacements for isoparametric LST elements using shape function interpolation
        X: Coordinate matrix
        T: Topology matrix
        Df: dof matrix
        V: displacement vector
        nel: number of elements
        dispFac: Displacement factor for scaling plot
        steps: Number of interpolation steps per edge
    '''

    plt.figure(figsize=[12, 8])
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')

    for el in range(1, nel + 1):
        eli = el - 1
        de = Df[eli, :]
        dei = de - 1
        Ve = V[dei].reshape(-1, 2)  # (6, 2)

        nodes = T[eli]
        Xe = X[nodes - 1]  # (6, 2)

        for i, idx in enumerate([[0, 5, 1], [1, 3, 2], [2, 4, 0]]):
            coords_undeformed = []
            coords_deformed = []
            for t in np.linspace(0, 1, steps):
                if i == 0:
                    z = [t, 1 - t, 0]
                elif i == 1:
                    z = [0, t, 1 - t]
                else:
                    z = [1 - t, 0, t]

                N = shape_funcs_LST_iso(z)
                x_und = N[idx] @ Xe[idx, 0]
                y_und = N[idx] @ Xe[idx, 1]
                x_def = x_und + dispFac * (N[idx] @ Ve[idx, 0])
                y_def = y_und + dispFac * (N[idx] @ Ve[idx, 1])

                coords_undeformed.append([x_und, y_und])
                coords_deformed.append([x_def, y_def])

            coords_undeformed = np.array(coords_undeformed)
            coords_deformed = np.array(coords_deformed)
            plt.plot(coords_undeformed[:, 0], coords_undeformed[:, 1], 'b:', linewidth=0.5)
            plt.plot(coords_deformed[:, 0], coords_deformed[:, 1], 'k-', linewidth=0.7)

    plt.title("Displacements - Scaling: " + str(dispFac))


def plotS_4x(X,T,Df,G,V,nel,s):
    '''
    Plot generelized stresses
        X: Coordinate matrix
        T: Topology matrix
        Df: dof matrix
        G: material matrix
        V: displacement vector
        nel: number of elements
        dL: Domain load
        s: stress plotted. 1 = normal force, 2 = shear force, 3 = moment 
        
        ????sFac: stress factor for scaling plot
    '''
    if s==1:
        Sec = '\u03C3xx' # Defining generelized stress name   
    elif s == 2:
        Sec = '\u03C3yy' # Defining generelized stress name   
    else:
        Sec = '\u03C4xy' # Defining generelized stress name   

    si = s-1
    fig = plt.figure(figsize=[12,8]) # Start figure
    plt.axis('equal') # set axis equal
    plt.xlabel('x') # Label x-axis
    plt.ylabel('y') # Label y-axis
    Ss = np.zeros((nel,6))
    for el in range(1,nel+1):

        eli = el-1 # element index
        #nodes
        no1, no2, no3 = T[eli,0], T[eli,1], T[eli,2] # first, second, third
        # index
        no1i, no2i, no3i = no1-1, no2-1, no3-1 # node 1, node 2, node 3
        # coordinates
        X1 = X[no1i]  # coordinates 1
        X2 = X[no2i]  # coordinates 2
        X3 = X[no3i]  # coordinates 3
        Ge = G[eli] # Material properties
        de = Df[eli,:] # index array for dofs
        dei = de-1 # dof index
        Ve = V[dei] # displacement vector

        # generelized stresses at nodes:
        Ss_el = np.zeros(6)
        for i,z in enumerate([[1,0,0], [0,1,0], [0,0,1], [0,1/2,1/2], [1/2,0,1/2], [1/2,1/2,0]]): #[0,1/2,1/2], [1/2,0,1/2], [1/2,1/2,0]
            sig = element.S(X1,X2,X3,Ge,Ve,z) #Generalised stresses: xx, yy, xy
            Ss_el[i] = sig[si] #Calculating stress at each node (and choosing xx, yy or xy via "si")

        Ss[eli] = Ss_el #Storing stress-values at each node in matrix with 6 columns (col1 = node 1)
            
    # normalise stresses
    #normalize = mpl.colors.Normalize(vmin=min(np.min(Ss),0),\
     #                                vmax=max(np.max(Ss),0)) 
    
    fac = 1
    vminp = min(np.min(Ss),0) * fac
    vmaxp = max(np.max(Ss),0) * fac
    # vminp = -1.2e4
    # vmaxp = 1.2e4

    for el in range(1,nel+1):
        eli = el-1 # element index
     
        norm = mpl.colors.Normalize(vmin=vminp, vmax=vmaxp)

        no1, no2, no3 = T[eli,0], T[eli,1], T[eli,2] # first, second, third
        no4, no5, no6 = T[eli,3], T[eli,4], T[eli,5] # fourth, fifth, sixth
        # index
        no1i, no2i, no3i = no1-1, no2-1, no3-1 # node 1, node 2, node 3
        no4i, no5i, no6i = no4-1, no5-1, no6-1 # node 4, node 5, node 6

        coordinates = np.array([X[no1i],X[no2i],X[no3i],X[no4i],X[no5i],X[no6i]])


        for i in [[0,5,4],[5,1,3],[4,3,2],[5,3,4]]:
            # Values at each vertex
            values = np.array([Ss[eli][i[0]],Ss[eli][i[1]],Ss[eli][i[2]]])
            # Triangle vertex coordinates
            points = np.array([coordinates[i[0]], coordinates[i[1]], coordinates[i[2]]])
            # Create triangulation
            triang = tri.Triangulation(points[:, 0], points[:, 1], [[0,1,2]])

            # fill element based on stress size
            # Inside your plotting loop:
            plt.tripcolor(triang, values, shading='gouraud', cmap='turbo', norm=norm)

    # displaying the title
    plt.title("Stresses: "+Sec)

     # add colourbar
    cbax = fig.add_axes([0.92, 0.12, 0.02, 0.78])
    cb = plt.colorbar(mpl.cm.ScalarMappable(cmap='turbo', norm=mpl.colors.Normalize(vmin=vminp, vmax=vmaxp)), cax=cbax)
    cb.set_label(Sec, rotation=270, labelpad=15)

def plotS_4x_iso(X, T, Df, G, V, nel, s):
    '''
    Plot generalized stresses on isoparametric LST:
      X:     node coordinates (n_nodes × 2)
      T:     element connectivity (nel × 6)
      Df:    dof array (nel × 12)
      G:     material params (nel × 3)
      V:     global displacement vector
      nel:   number of elements
      s:     which stress: 1 = normal force, 2 = shear force, 3 = moment
    '''
    # pick symbol
    if   s == 1: Sec = r'$\sigma_{xx}$'
    elif s == 2: Sec = r'$\sigma_{yy}$'
    else:         Sec = r'$\tau_{xy}$'
    si = s-1

    # --- exactly the same fig/ax setup as plotS_4x ---
    fig, ax = plt.subplots(figsize=[9,6])
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # axis limits with 0.5 padding
    x_min = np.floor(np.min(X[:,0]))
    x_max = np.ceil (np.max(X[:,0]))
    y_min = np.floor(np.min(X[:,1]))
    y_max = np.ceil (np.max(X[:,1]))
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)

    # ticks every 2 units
    tick = 2
    ax.set_xticks(np.arange(x_min, x_max+0.1, tick))
    ax.set_yticks(np.arange(y_min, y_max+0.1, tick))

    # --- compute stresses at the six nodes of each element (unchanged) ---
    Ss = np.zeros((nel,6))
    for el in range(1, nel+1):
        eli = el-1
        no1, no2, no3 = T[eli,0], T[eli,1], T[eli,2]
        no1i, no2i, no3i = no1-1, no2-1, no3-1
        X1, X2, X3 = X[no1i], X[no2i], X[no3i]
        Ge = G[eli]
        de = Df[eli,:] - 1
        Ve = V[de]
        Ss_el = np.zeros(6)
        for i,z in enumerate([[1,0,0],[0,1,0],[0,0,1],[0,1/2,1/2],[1/2,0,1/2],[1/2,1/2,0]]):
            sig = element.S(X1, X2, X3, Ge, Ve, z)
            Ss_el[i] = sig[si]
        Ss[eli] = Ss_el

    # normalize exactly like plotS_4x (fac=0.05)
    fac   = 1
    vminp = min(np.min(Ss), 0) * fac
    vmaxp = max(np.max(Ss), 0) * fac

    # draw each sub‐triangle
    norm = mpl.colors.Normalize(vmin=vminp, vmax=vmaxp)
    for el in range(1, nel+1):
        eli = el-1
        coords = np.array([X[T[eli,i]-1] for i in range(6)])
        for tri_idx in ([0,5,4],[5,1,3],[4,3,2],[5,3,4]):
            pts    = coords[tri_idx]
            vals   = Ss[eli][tri_idx]
            triang = tri.Triangulation(pts[:,0], pts[:,1], [[0,1,2]])
            ax.tripcolor(triang, vals, shading='gouraud', cmap='turbo', norm=norm)

    # draw element outlines exactly as in plotS_4x
    for el in range(nel):
        nodes = [T[el,i]-1 for i in range(6)]
        polyx = [X[nodes[i],0] for i in [0,5,1,3,2,4,0]]
        polyy = [X[nodes[i],1] for i in [0,5,1,3,2,4,0]]
        ax.plot(polyx, polyy, color='black', linewidth=0.4, alpha=0.34)

    ax.set_title(f"Stresses: {Sec}")

    # --- same divider/colorbar setup ---
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="3%", pad=0.15)
    cbar    = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='turbo'),
                           cax=cax, orientation='vertical')
    # ten ticks, scientific notation
    ticks = np.linspace(vminp, vmaxp, 10)
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))
    cbar.set_label(Sec, rotation=270, labelpad=15)

    plt.tight_layout()