import numpy as np
import LST as element
import solvers as solvers
import postLST as post
import matplotlib.pyplot as plt

from generate_mesh_simple import generate_LST_mesh

import datetime
import sys

# Logging of output-data
logfile = open("output.txt", "w", encoding="utf-8")
sys.stdout = logfile  
sys.stderr = logfile  
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logfile.write(f"Timestamp: {timestamp}\n")
logfile.write("FEMaztec LST\n\n")


# Parameters
L = 8          # Length       [m]
H = 2          # Height       [m]
t  = 0.17      # Thickness    [m]
E  = 2.0e11    # Youngs mod.  [N/m2]
P  = 1.34e5    # Point load   [N]
nu = 0.3       # Poisson's ratio

# Factors for plotting 
dFac = 1e3     # displacement plot factor
sFac = 1e-6    # principal stress plot factors

# Topology
X, T = generate_LST_mesh(L,H,36,9)
nno  = len(X)  # Number of nodes
nel  = len(T)  # Number of elements

# Material matrix 
G = np.tile([E, nu, t], (nel, 1))

# Supports: (global dof, value)
U = np.array([
     (1,0),    
     (658*2,0),
     (1315*2-1,0),
     ])

# Point loads: (global_dof, value)
pL = np.array([
])

# Boundary loads: (element, side, x_start, x_end, y_start, y_end)
mmt = 6*P*L/(H**2) # Moment
bL = np.array([
    # Left side shear
    (577,2,0,0,0,50*P/(81*H)),
    (505,2,0,0,50*P/(81*H),84*P/(81*H)), 
    (433,2,0,0,84*P/(81*H),110*P/(81*H)), 
    (361,2,0,0,110*P/(81*H),120*P/(81*H)), 
    (289,2,0,0,121*P/(81*H),121*P/(81*H)), 
    (217,2,0,0,120*P/(81*H),110*P/(81*H)), 
    (145,2,0,0,110*P/(81*H),84*P/(81*H)), 
    (73,2,0,0,84*P/(81*H),50*P/(81*H)), 
    (1,2,0,0,50*P/(81*H),0), 

    # Right side shear
    (72, 1,0,0,0,-50*P/(81*H)),
    (144, 1,0,0,-50*P/(81*H),-28*P/(27*H)),
    (216,1,0,0,-28*P/(27*H),-110*P/(81*H)),
    (288,1,0,0,-110*P/(81*H),-40*P/(27*H)), 
    (360,1,0,0,-121*P/(81*H),-121*P/(81*H)), 
    (432,1,0,0,-120*P/(81*H),-110*P/(81*H)), 
    (504,1,0,0,-110*P/(81*H),-84*P/(81*H)), 
    (576,1,0,0,-28*P/(27*H),-50*P/(81*H)), 
    (648,1,0,0,-50*P/(81*H),0), 

     # Moment
    (577,2,-mmt,-7/9*mmt,0,0),
    (505,2,-7/9*mmt,-5/9*mmt,0,0),
    (433,2,-5/9*mmt,-3/9*mmt,0,0),
    (361,2,-3/9*mmt,-1/9*mmt,0,0),
    (289,2,-1/9*mmt,1/9*mmt,0,0),
    (217,2,1/9*mmt,3/9*mmt,0,0),
    (145,2,3/9*mmt,5/9*mmt,0,0),
    (73,2,5/9*mmt,7/9*mmt,0,0),
    (1,2,7/9*mmt,mmt,0,0),
     ])

# domain load (el, px, py)
dL = np.array([
     ])

# Program
# -----------------------------------------------------------------------------
Df = element.buildDf(T, nel) # Dof matrix
K  = element.buildK(X, T, G, Df, nel) # stiffnessmatrix, K
R  = element.buildR(X, T, Df, pL, bL, dL) # load vector, R 
V, Ru  = solvers.linearLST(K,R,U) # solve system equations for displacements

# Post processing
# -----------------------------------------------------------------------------
plt.rcParams['figure.dpi']=110 # set resolution of plots

stress_type = 1 # xx=1 , yy=2 , xy=3

post.listV(V,Df) # List displacements
post.listS(X,T,Df,G,V,nel,stress_type) # List stresses
post.listR(U,R,Ru) # List reactions
post.plotTop(X,T,nno,nel,U,False,True) # Plot topology (...., plot_no:bool , plot_supp:bool)
post.plotDof(X,T,Df,V,nel,dFac) # plot deformed structure
post.plotS_4x(X,T,Df,G,V,nel,stress_type) # Plot normal force
plt.show() # show plots

logfile.close()