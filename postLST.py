import numpy as np
import LST as element

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as tri
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def listV(V,Df): 
    '''list displacements
    # V: displacement vector
    Df: Dof matrix
    '''

    nel = len(Df) # number of element
    print(' ')
    print('Displacements (V):')
    print('-------------------------------------------------------------------------------'+\
          '--------------------------------------------------------------------------------')
    print(' el   Vx1          Vy1          Vx2          Vy2          Vx3          Vy3          '+\
                'Vx4          Vy4          Vx5          Vy5          Vx6          Vy6')


    for el in range(1,nel+1):
        eli = el-1 # element index
        de = Df[eli,:] # index array for dofs
        dei = de-1 # node 1 index
        Ve = V[dei] # element displacement vector
        print(('{n:3} {Vx1:12.4e} {Vy1:12.4e} {Vx2:12.4e} {Vy2:12.4e}'+\
                    ' {Vx3:12.4e} {Vy3:12.4e} {Vx4:12.4e} {Vy4:12.4e}'+\
                    ' {Vx5:12.4e} {Vy5:12.4e} {Vx6:12.4e} {Vy6:12.4e}')
              .format(n=el,Vx1=Ve[0], Vy1=Ve[1], Vx2=Ve[2], Vy2=Ve[3],\
                           Vx3=Ve[4], Vy3=Ve[5], Vx4=Ve[6], Vy4=Ve[7],\
                           Vx5=Ve[8], Vy5=Ve[9], Vx6=Ve[10], Vy6=Ve[11]))
        
    print('-------------------------------------------------------------------------------'+\
          '--------------------------------------------------------------------------------')

def listEps(X,T,Df,V,nel):
    '''
    list generalized strains
        X: Coordinate matrix
        T: Topology matrix
        Df: dof matrix
        V: displacement vector
        nel: number of elements
    '''
    nel = len(Df) # number of element

    print(' ')
    print(f'Strains at each  node (number above strains is the global number of node):')
    print('---------------------------------------------------------------------------------')

    print(' el              no1            no2            no3            no4            no5            no6')

    for el in range(1,nel+1):
        eli = el-1 # element index
        #nodes
        no1, no2, no3 = T[eli,0], T[eli,1], T[eli,2] #first, second, third   
        no4, no5, no6 = T[eli,3], T[eli,4], T[eli,5] #fourth, fifth, sixth  
        # index
        no1i, no2i, no3i = no1-1, no2-1, no3-1 # node 1, 2 & 3
        # coordinates
        X1, X2, X3 = X[no1i], X[no2i], X[no3i]  # node 1, 2 & 3

        de = Df[eli,:] # index array for dofs
        dei = de-1 # dof index
        Ve = V[dei] # displacement vector

        epsilon = np.zeros([3,6])
        for i,z in enumerate([[1,0,0],[0,1,0], [0,0,1], [0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]]):
            epsilon[:,i] = element.Eps(X1,X2,X3,Ve,z) # Calculating s_xx, s_yy & s_xy at each node and saving as column in 3x6-matrix

       # print(('{n:4} {gnd:6} {nd1:^12d} {nd2:^12d} {nd3:^12d} {nd4:^12d} {nd5:^12d} {nd6:^12d}')
       #        .format(n='',gnd='Glno:',nd1=no1, nd2=no2, nd3=no3, nd4=no4,nd5=no5, nd6=no6)) # Listing global number of node
        print(('{n:4} {gnd:6} {nd1:^14d} {nd2:^14d} {nd3:^14d} {nd4:^14d} {nd5:^14d} {nd6:^14d}')
              .format(n='',gnd='',nd1=no1, nd2=no2, nd3=no3, nd4=no4,nd5=no5, nd6=no6)) # Listing global number of node
        # sig_xx
        print(('{sig:<5} {s__:4} {st1:14.6e} {st2:14.6e} {st3:14.6e} {st4:14.6e} {st5:14.6e} {st6:14.6e}')
              .format(sig=el,s__='\u03B5xx:',st1=epsilon[0,0], st2=epsilon[0,1], st3=epsilon[0,2], st4=epsilon[0,3],st5=epsilon[0,4], st6=epsilon[0,5]))
        #sig_yy
        print(('{sig:5} {s__:4} {st1:14.6e} {st2:14.6e} {st3:14.6e} {st4:14.6e} {st5:14.6e} {st6:14.6e}')
              .format(sig='',s__='\u03B5xx:',st1=epsilon[1,0], st2=epsilon[1,1], st3=epsilon[1,2], st4=epsilon[1,3],st5=epsilon[1,4], st6=epsilon[1,5]))
        #sig_xy
        print(('{sig:5} {s__:4} {st1:14.6e} {st2:14.6e} {st3:14.6e} {st4:14.6e} {st5:14.6e} {st6:14.6e}')
              .format(sig='',s__='\u03B3xy:',st1=epsilon[2,0], st2=epsilon[2,1], st3=epsilon[2,2], st4=epsilon[2,3],st5=epsilon[2,4], st6=epsilon[2,5]))
        #print('') # Spacing  
    print('---------------------------------------------------------------------------------')

def listS(X,T,Df,G,V,nel,s):
    '''
    list generalized stresses
        X: Coordinate matrix
        T: Topology matrix
        Df: dof matrix
        G: material matrix
        V: displacement vector
        nel: number of elements
        s: stress plotted. 1 = normal force, 2 = shear force, 3 = moment #! Normal force, shear and moment correct?
    '''
    nel = len(Df) # number of element

    if s==1:
        Sec = '\u03C3xx' # Defining generelized stress name   
    elif s == 2:
        Sec = '\u03C3yy' # Defining generelized stress name   
    else:
        Sec = '\u03C4xy' # Defining generelized stress name   

    print(' ')
    print(f'Stresses at each  node ({Sec}) (number above stress is the global number of node):')
    print('---------------------------------------------------------------------------------')

    print(' el   no1          no2          no3          no4          no5          no6')

    for el in range(1,nel+1):
        eli = el-1 # element index
        #nodes
        no1, no2, no3 = T[eli,0], T[eli,1], T[eli,2] #first, second, third   
        no4, no5, no6 = T[eli,3], T[eli,4], T[eli,5] #fourth, fifth, sixth  
        # index
        no1i, no2i, no3i = no1-1, no2-1, no3-1 # node 1, 2 & 3
        # coordinates
        X1, X2, X3 = X[no1i], X[no2i], X[no3i]  # node 1, 2 & 3

        Ge = G[eli] # Material properties
        de = Df[eli,:] # index array for dofs
        dei = de-1 # dof index
        Ve = V[dei] # displacement vector

        sigma = np.zeros(6)
        for i,z in enumerate([[1,0,0],[0,1,0], [0,0,1], [0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]]):
            sigma[i] = element.S(X1,X2,X3,Ge,Ve,z)[s-1]

        # print(('{n:3} {nd1:12d} {nd2:12d} {nd3:12d} {nd4:12d} {nd5:12d} {nd6:12d}')
        #       .format(n='',nd1=no1, nd2=no2, nd3=no3, nd4=no4,nd5=no5, nd6=no6))

        # print(('{n:3} {st1:12.4e} {st2:12.4e} {st3:12.4e} {st4:12.4e} {st5:12.4e} {st6:12.4e}')
        #       .format(n=el,st1=sigma[0], st2=sigma[1], st3=sigma[2], st4=sigma[3],st5=sigma[4], st6=sigma[5]))

        print(('{n:3} {nd1:12d} {nd2:22d} {nd3:22d} {nd4:22d} {nd5:22d} {nd6:22d}')
              .format(n='',nd1=no1, nd2=no2, nd3=no3, nd4=no4,nd5=no5, nd6=no6))
        
        print(('{n:3} {st1:12.15e} {st2:12.15e} {st3:12.15e} {st4:12.15e} {st5:12.15e} {st6:12.15e}')
              .format(n=el,st1=sigma[0], st2=sigma[1], st3=sigma[2], st4=sigma[3],st5=sigma[4], st6=sigma[5]))
        
    print('---------------------------------------------------------------------------------')
    
def listS_total(X,T,Df,G,V,nel):
    '''
    list generalized stresses (more decimals)
        X: Coordinate matrix
        T: Topology matrix
        Df: dof matrix
        G: material matrix
        V: displacement vector
        nel: number of elements
    '''
    nel = len(Df) # number of element

    print(' ')
    print(f'Stresses at each  node (number above stresses is the global number of node):')
    print('---------------------------------------------------------------------------------')

    print(' el               no1            no2            no3            no4            no5            no6')

    for el in range(1,nel+1):
        eli = el-1 # element index
        #nodes
        no1, no2, no3 = T[eli,0], T[eli,1], T[eli,2] #first, second, third   
        no4, no5, no6 = T[eli,3], T[eli,4], T[eli,5] #fourth, fifth, sixth  
        # index
        no1i, no2i, no3i = no1-1, no2-1, no3-1 # node 1, 2 & 3
        # coordinates
        X1, X2, X3 = X[no1i], X[no2i], X[no3i]  # node 1, 2 & 3

        Ge = G[eli] # Material properties
        de = Df[eli,:] # index array for dofs
        dei = de-1 # dof index
        Ve = V[dei] # displacement vector

        sigma = np.zeros([3,6])
        for i,z in enumerate([[1,0,0],[0,1,0], [0,0,1], [0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]]):
            sigma[:,i] = element.S(X1,X2,X3,Ge,Ve,z) # Calculating s_xx, s_yy & s_xy at each node and saving as column in 3x6-matrix

        print(('{n:5} {gnd:6} {nd1:^14d} {nd2:^14d} {nd3:^14d} {nd4:^14d} {nd5:^14d} {nd6:^14d}')
              .format(n='',gnd='',nd1=no1, nd2=no2, nd3=no3, nd4=no4,nd5=no5, nd6=no6)) # Listing global number of node

        # sig_xx
        print(('{sig:<5} {s__:4} {st1:14.10e} {st2:14.10e} {st3:14.10e} {st4:14.10e} {st5:14.10e} {st6:14.10e}')
              .format(sig=el,s__='\u03C3xx:',st1=sigma[0,0], st2=sigma[0,1], st3=sigma[0,2], st4=sigma[0,3],st5=sigma[0,4], st6=sigma[0,5]))

        #sig_yy
        print(('{sig:5} {s__:4} {st1:14.10e} {st2:14.10e} {st3:14.10e} {st4:14.10e} {st5:14.10e} {st6:14.10e}')
              .format(sig='',s__='\u03C3yy:',st1=sigma[1,0], st2=sigma[1,1], st3=sigma[1,2], st4=sigma[1,3],st5=sigma[1,4], st6=sigma[1,5]))
        
        #sig_xy
        print(('{sig:5} {s__:4} {st1:14.10e} {st2:14.10e} {st3:14.10e} {st4:14.10e} {st5:14.10e} {st6:14.10e}')
              .format(sig='',s__='\u03C3xy:',st1=sigma[2,0], st2=sigma[2,1], st3=sigma[2,2], st4=sigma[2,3],st5=sigma[2,4], st6=sigma[2,5]))
        
    print('---------------------------------------------------------------------------------')

def listR(U,R,Ru):
    '''
    List reactions
        U: Supports
        R: Load vector
        Ru: nodal forces of supported dofs
    '''
    print(' ')
    print('Reactions (R):')
    Lines = '----------------'
    print(Lines)
    print(' R     Value')
    dof = U[:,0].astype('i4') # supported dofs
    dofi = dof-1 # index of supported dofs
    Rs = -R[dofi]+Ru # add load to nodal forces of supported dofs
    
    for i in range (len(Rs)):
        dof  = U[i,0] # dof
        node = np.ceil(dof / 2).astype(int) # node
        if dof % 2 == 1: # if the dof is one more than a multiple of 2
            Init = str(' R'+str(node)+'x ') # x-direction
        else: # else
            Init = str(' R'+str(node)+'y ') # y direction
        Value = '{R:12.4e}'.format(R=Rs[i])
        print(Init+Value)
    print(Lines)
            

def plotTop(X,T,nno,nel,U,plot_no:bool,plot_supp:bool):
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
    if plot_no:
        for no in range(1,nno+1):
            noi = no-1 #  node index
            # plot node number
            plt.text(X[noi,0],X[noi,1],str(no),\
                    bbox={'boxstyle':'round','facecolor':'lightgray', \
                        'alpha':0.5,'pad':0.1},\
                    horizontalalignment='center',verticalalignment='center',fontsize=6) 
    for el in range(1,nel+1):
        # peri = np.array([0,1,2]).astype('i4')
        eli = el-1 # element index
        #nodes
        no1 = T[eli,0] # first
        no2 = T[eli,1] # second
        no3 = T[eli,2] # third
        # index
        no1i = no1-1 # node 1
        no2i = no2-1 # node 2
        no3i = no3-1 # node 3
        # coordinates
        X1 = [X[no1i,0], X[no2i,0], X[no3i,0], X[no1i,0]] # x-coord
        Y1 = [X[no1i,1], X[no2i,1], X[no3i,1], X[no1i,1]] # y-coord
        plt.plot(X1,Y1,'b-',linewidth=0.5) # plot the 
        Xm = np.mean([X[no1i,0], X[no2i,0], X[no3i,0]]) # x-coord
        Ym = np.mean([X[no1i,1], X[no2i,1], X[no3i,1]]) # y-coord
        # plot element number
        plt.text(Xm,Ym,str(el),\
                 horizontalalignment='center',verticalalignment='center',fontsize=4) 
    # displaying the title

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

    plt.title("Topology")

def plotDof(X,T,Df,V,nel,dispFac): 
    '''
    Plot displacements
        X: Coordinate matrix
        T: Topology matrix
        Df: dof matrix
        V: displacement vector
        nel: number of elements
        dispFac: Displacement factor for scaling plot
    '''
    plt.figure(figsize=[6,4]) # Start figure
    plt.axis('equal') # set axis equal
    plt.xlabel('x') # Label x-axis
    plt.ylabel('y') # Label y-axis
    for el in range(1,nel+1):
        eli = el-1 # element index
        de = Df[eli,:] # index array for dofs
        dei = de-1 # dof index
        Ve = V[dei] # displacement vector
        #nodes
        no1, no2, no3 = T[eli,0], T[eli,1], T[eli,2] #first, second, third   
        no4, no5, no6 = T[eli,3], T[eli,4], T[eli,5] #fourth, fifth, sixth  
        # index
        no1i, no2i, no3i = no1-1, no2-1, no3-1 # node 1, 2 & 3
        no4i, no5i, no6i = no4-1, no5-1, no6-1 # node 4, 5 % 6
        # coordinates
        X1 = np.array([X[no1i,0], X[no6i,0], X[no2i,0], X[no4i,0], X[no3i,0], X[no5i,0], X[no1i,0]]) # x
        Y1 = np.array([X[no1i,1], X[no6i,1], X[no2i,1], X[no4i,1], X[no3i,1], X[no5i,1], X[no1i,1]]) # y

        # plot the element
        plt.plot(X1,Y1,'b:',linewidth=0.5) 
        # transformation matrix and element length
        X1u = X1+dispFac*np.array(Ve[[0,10,2,6,4,8,0]]) # x
        Y1u = Y1+dispFac*np.array(Ve[[1,11,3,7,5,9,1]]) # y
        plt.plot(X1u,Y1u,'k-',linewidth=0.5) # plot the element¨
        plt.xticks(np.arange(np.floor(np.min(X[:,0])), np.ceil(np.max(X[:,0]))+2, 2))
        plt.yticks(np.arange(np.floor(np.min(X[:,1])), np.ceil(np.max(X[:,1]))+2, 2))
    # displaying the title
    plt.title("Displacements - Scaling: {:.2e}".format(dispFac))


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
        s: stress plotted: 1=sigma_xx, 2=sigma_yy, 3=tau_xy
    '''
    if s==1:
        Sec = '\u03C3xx' # Defining generelized stress name   
    elif s == 2:
        Sec = '\u03C3yy' # Defining generelized stress name   
    else:
        Sec = '\u03C4xy' # Defining generelized stress name   

    si = s-1

    fig, ax = plt.subplots(figsize=[6,4])
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_xlim(np.floor(np.min(X[:,0])) - 0.5, np.ceil(np.max(X[:,0])) + 0.5)
    ax.set_ylim(np.floor(np.min(X[:,1])) - 0.5, np.ceil(np.max(X[:,1])) + 0.5)

    # Generate tick positions every 0.5 in x-direction
    tick = 2
    # For x-axis: ticks every 2, but with a 1 gap at each end
    x_min = np.floor(np.min(X[:,0]))
    x_max = np.ceil(np.max(X[:,0]))
    x_ticks = np.arange(x_min, x_max+0.1, tick)
    ax.set_xticks(x_ticks)

    # For y-axis: ticks every 2, but with a 1 gap at each end
    y_min = np.floor(np.min(X[:,1]))
    y_max = np.ceil(np.max(X[:,1]))
    y_ticks = np.arange(y_min, y_max+0.1, tick)
    ax.set_yticks(y_ticks)

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
            
    fac = 1
    vminp = min(np.min(Ss),0) * fac
    vmaxp = max(np.max(Ss),0) * fac

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
    
    # Plot LST element outlines
    for el in range(nel):
        no1, no2, no3 = T[el, 0]-1, T[el, 1]-1, T[el, 2]-1
        no4, no5, no6 = T[el, 3]-1, T[el, 4]-1, T[el, 5]-1
        x_coords = [X[no1, 0], X[no6, 0], X[no2, 0], X[no4, 0],
                     X[no3, 0], X[no5, 0], X[no1, 0]]
        y_coords = [X[no1, 1], X[no6, 1], X[no2, 1], X[no4, 1],
                     X[no3, 1], X[no5, 1], X[no1, 1]]
        ax.plot(x_coords, y_coords, color='black', linewidth=0.4, alpha=0.34, linestyle='-')

    ax.set_title("Stresses: " + Sec)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)

    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='turbo'),
                        cax=cax, orientation='vertical')
    n_ticks = 10
    ticks = np.linspace(vminp, vmaxp, n_ticks)
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))
