
import numpy as np
import matplotlib.pyplot as plt

import LST as element
import solvers as solvers
import postLST as post
import ITC as ITC

# Function for generating mesh
from utils.generate_mesh_simple import generate_LST_mesh

# Parameters
L = 20          # Length       [m]
H = 10          # Height       [m]
t  = 1.0       # Thickness    [m]
E  = 2.1e11    # Youngs mod.  [N/m2]
P  = 1e4       # Point load   [N/m2]
nu = 0.3       # Poisson's ratio

s_1ix = 2.3077e11 # [N/m2]
s_1iy = 0.6923e11 # [N/m2]

# Factors for plotting 
dFac = 1e7     # displacement plot factor
sFac = 1e-6    # principal stress plot factors

# Topology
X, T = generate_LST_mesh(L,H,40,20)
nno  = len(X)  # Number of nodes
nel  = len(T)  # Number of elements

# Material matrix 
G = np.tile([E, nu, t], (nel, 1))

# Supports: (global dof, value)
U = [(2,0)]
for i in range (2,42*2,2): # Step = 2 because of 2 dofs per node
    U.append((81*i-1, 0))  # Supports in x-dofs
    U.append((81*i, 0))    # Supports in y-dofs
U = np.array(U)

# Point load (global_dof, value)
pL = np.array([])
dL =np.array([])

# Boundary loads of System 0: (element, side, x_start, x_end, y_start, y_end)
bL0 = np.array([
    # Left side shear
    (1557,1,0,0,-P,-P),
    (1559,1,0,0,-P,-P),
    (1561,1,0,0,-P,-P),  
    (1563,1,0,0,-P,-P),
     ])

# Boundary loads of System 1: (element, side, x_start, x_end, y_start, y_end)
bL1 = np.array([
    # Bottom
    (40,3,0,0,-s_1iy,-s_1iy),
    (42,3,0,0,-s_1iy,-s_1iy),
    # Right side
    (42,1,s_1ix,s_1ix,0,0),
    (122,1,s_1ix,s_1ix,0,0),
    #Top  
    (121,1,0,0,s_1iy,s_1iy),
    (119,1,0,0,s_1iy,s_1iy),
    #Left side
    (119,2,-s_1ix,-s_1ix,0,0),
    (39,2,-s_1ix,-s_1ix,0,0),
     ])

# Program
# -----------------------------------------------------------------------------
Df = element.buildDf(T, nel) # Dof matrix

# System 0
K0  = element.buildK(X, T, G, Df, nel) # stiffnessmatrix, K
R0  = element.buildR(X, T, Df, pL, bL0, dL) # load vector, R 
V0, Ru0  = solvers.linearLST(K0,R0,U) # solve system equations for displacements

# System 1
K1  = element.buildK(X, T, G, Df, nel) # stiffnessmatrix, K
R1  = element.buildR(X, T, Df, pL, bL1, dL) # load vector, R 
V1, Ru1  = solvers.linearLST(K1,R1,U) # solve system equations for displacements

ITC_val = ITC.ITC_S0_E1(X,T,G,Df,V0,V1,t)
#Factor for scaling of plot
fac = 0.017

# # Plotting geometry and topology
post.plotTop(X,T,nno,nel,U,False,True)

# #FEM-plots of System 0
post.plotDof(X,T,Df,V0,nel,dFac)
post.plotS_4x(X,T,Df,G,V0,nel,1)

# #FEM-plots of System 1
post.plotDof(X,T,Df,V1,nel,dFac)
post.plotS_4x(X,T,Df,G,V1,nel,1)

#ITC-plot
ITC.plotITC_4x(X,T,nel,ITC_val,fac,False)
plt.show()