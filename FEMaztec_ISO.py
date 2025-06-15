import numpy as np
import LST as element
import LST_iso as element_iso
import postLST as post
import postLST_iso as post_iso
import solvers as solvers
import matplotlib.pyplot as plt

import datetime
import sys

from mesh_iso_total_reordered import generate_wall_with_hole
from mesh_iso_total_reordered import generate_wall_with_hole_fixed_elements
from mesh_iso_total_reordered import generate_wall_with_hole_uniform_mesh

# Logfile
logfile = open("output.txt", "w", encoding="utf-8")
sys.stdout = logfile  # Redirect standard output to file
sys.stderr = logfile  # (Optional) Redirect errors too
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logfile.write(f"Timestamp: {timestamp}\n")
logfile.write("Program til statisk beregning af 2D skivekonstruktioner\n\n")

np.set_printoptions(precision=3, suppress=False, threshold=1000)

#! ----------------------------------------------------------------------------------------------------------------------------------------------------------------

L_ = 50        # Length       [m]
H_ = 50        # Height       [m]
R_ = 1         # Radius       [m]
t  = 0.17      # Thickness    [m]
E  = 2.1e11    # Youngs mod.  [N/m2]
p  = 1.34e5    # Load   [N/m]
nu = 0.3       # Poisson's ratio

dFac = 1e2 # displacement plot factor

#X_cnode, T = generate_wall_with_hole(L=L_, H=H_ ,R=R_, lc_outer=2,lc_inner=1)
#X_cnode, T = generate_wall_with_hole_fixed_elements(L=H_, H=H_ ,R=R_,target_elements=400)
X_cnode, T = generate_wall_with_hole_uniform_mesh(L=H_, H=H_ ,R=R_,mesh_size=5)

# Removing center node from pygmsh
used = np.unique(T.flatten())
expected = np.arange(used.min(), used.max() + 1) # Find the full expected range of node indices
missing = np.setdiff1d(expected, used) # Find missing ones
new_indices = {old: new for new, old in enumerate(used)}
T[T>missing[0]] -= 1

T += 1
X = X_cnode[used]

nno  = len(X)  # Number of nodes
nel  = len(T) # Number of elements

G = np.tile([E, nu, t], (nel, 1))

top_node = 4
left_boundary_nodes = [66,67,68,75,76,77,78,79,80,81,82,83,84,1,74,75,76,77,78,79,80,70,71,72,73,69]

# collect all the DOF‐indices you want to fix:
dofs = []
for n in left_boundary_nodes:
    dofs += [2*n-1]      # u_x at node n is DOF 2n-1, u_y at node n is DOF 2n

# and don’t forget the two DOFs of your top node:
dofs += [2*top_node-1, 2*top_node]

# now make your U‐array with a zero prescribed value in the second column:
U = np.array([(d, 0) for d in dofs], dtype=int)

pL = np.array([
    ])

right_boundary_elements = [294,362,234,133,116,255,139,198,230,287,473]

# say all these are on side=1, with these q‐values:
side = 1
qx_start, qx_end = p, p
qy_start, qy_end = 0, 0    # just as an example

bL = np.array([
    (elem, side, qx_start, qx_end, qy_start, qy_end) for elem in right_boundary_elements], dtype=float)

dL = np.array([
     ])

# Program
# -----------------------------------------------------------------------------
Df = element.buildDf(T, nel) # Dof matrix
K  = element_iso.buildK_iso(X, T, G, Df, nel) # stiffnessmatrix, K
R  = element.buildR(X, T, Df, pL, bL, dL) # load vector, R 
V, Ru  = solvers.linearLST(K,R,U) # solve system equations for displacements

# Post processing
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi']=100 # set resolution of plots
stress_type = 1 # xx=1 , yy=2 , xy=3

post.listV(V,Df) # List displacements
post.listS(X,T,Df,G,V,nel,stress_type) # List stresses
post.listR(U,R,Ru) # List reactions
post_iso.plotTop_iso(X,T,nno,nel,U,False,True,steps=20)
post_iso.plotDof_iso(X,T,Df,V,nel,dFac,steps=20)

# Better implementations of plotS
post_iso.plotS_4x_iso(X,T,Df,G,V,nel,stress_type)

plt.show() # show plots

logfile.close()