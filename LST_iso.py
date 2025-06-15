import numpy as np
import LST as element

def buildK_iso(X,T,G,Df,nel):
    '''
    Returns system stiffness matrix
        X: Coordinate matrix
        T: Topology matrix
        G: Material matrix
        Df: dof matrix
        nel: number of elements
    '''
    
    K=np.zeros((np.max(Df),np.max(Df))) # init K
    for el in range(1,nel+1): 
        eli = el-1
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
                       X[no6i],])
        
        # element stiffness matrix 
        k = kLST_iso(Xe,G[eli]) 
        
        # array for dofs
        de = Df[eli,:] # dofs
        dei = de-1 # dof index
        # assembling K
        K[np.ix_(dei,dei)] += k 
    return K


def dNLST_dzeta_iso(z):
    '''
    Returns dNLST/dzeta matrix (7.4) for calculating Jacobian matrix
    z: array/list with barycentric coordinates; [z1, z2]
    '''
    dNLST_dzeta = np.array([[4*z[0]-1 , 0 , 4*z[0]+4*z[1]-3 , -4*z[1] , 4-8*z[0]-4*z[1] , 4*z[1]],
                           [0 , 4*z[1]-1 , 4*z[0]+4*z[1]-3 , 4-4*z[0]-8*z[1] , -4*z[0] , 4*z[0]]])

    return dNLST_dzeta


def JLST_iso(Xe,z):
    '''
    Returns the Jacobian matrix
    Xe: 6x2 array containg coordinates of all nodes in element
    '''
    # dNLST/dzeta matrix (7.4) for calculating Jacobian matrix
    dNdz = dNLST_dzeta_iso(z)

    # Calculating Jacobian matrix
    J = np.dot(dNdz,Xe)

    return J

def JLSTinverse_iso(J):
    '''
    Returns the inverse Jacobian matrix for an LST-element
    using the analytical 2x2 inverse formula (6.16)
    
    Parameters:
        J (ndarray): 2x2 Jacobian matrix

    Returns:
        Jinverse (ndarray): 2x2 inverse Jacobian matrix
    '''

    #Calculating deterimant of J
    detJ = np.linalg.det(J)


    Jinverse = (1 / detJ) * np.array([[ J[1,1], -J[0,1]],
                                      [-J[1,0],  J[0,0]]])

    return Jinverse

def BLST_iso(Xe,z):
    ''' 
    Parameteres:
        Xe: 6x2 matrix containing coordinates of each node
        z: Array with zeta-values (only z1 & z2)

    Returns:
        B: Strain interpolation matrix for isoparametric element
        J: Jacobian matrix for z=[z1,z2]
        
    '''
    # For easier use of zeta values
    z1 ,z2 = z[0], z[1]

    # Calculating Jinverse
    J = JLST_iso(Xe,z)
    Jinverse = JLSTinverse_iso(J)

    #Below functions are excluding 1/(2*A) - that is added when creating the B-matrix
    dN1dx = (4*z1-1)*Jinverse[0,0]
    dN1dy = (4*z1-1)*Jinverse[1,0]

    dN2dx = (4*z2-1)*Jinverse[0,1]
    dN2dy = (4*z2-1)*Jinverse[1,1]

    dN3dx = (4*z1 + 4*z2 - 3)*Jinverse[0,0] + (4*z1 + 4*z2 - 3)*Jinverse[0,1]
    dN3dy = (4*z1 + 4*z2 - 3)*Jinverse[1,0] + (4*z1 + 4*z2 - 3)*Jinverse[1,1]

    dN4dx = -4*z2*Jinverse[0,0] + (4-4*z1-8*z2)*Jinverse[0,1]
    dN4dy = -4*z2*Jinverse[1,0] + (4-4*z1-8*z2)*Jinverse[1,1]
    
    dN5dx = (4-8*z1-4*z2)*Jinverse[0,0] - 4*z1*Jinverse[0,1]
    dN5dy = (4-8*z1-4*z2)*Jinverse[1,0] - 4*z1*Jinverse[1,1]

    dN6dx = 4*z2*Jinverse[0,0] + 4*z1*Jinverse[0,1]
    dN6dy = 4*z2*Jinverse[1,0] + 4*z1*Jinverse[1,1]

    # strain interpolation matrix
    B_iso = np.array([
        (dN1dx , 0 , dN2dx , 0 , dN3dx , 0 , dN4dx , 0 , dN5dx , 0 , dN6dx , 0),
        (0 , dN1dy , 0 , dN2dy , 0 , dN3dy , 0 , dN4dy , 0 , dN5dy , 0 , dN6dy),
        (dN1dy , dN1dx , dN2dy , dN2dx , dN3dy , dN3dx , dN4dy , dN4dx , dN5dy , dN5dx , dN6dy , dN6dx),
        ])
    return B_iso, J


def kLST_iso(Xe, Ge): 
    '''
    Parameters:
        Xe: 6x2 array containing coordinates of all nodes in the element
        Ge: element material properties [E, nu, t]
    Returns:
        k: element stiffness matrix (12x12 for 6-node triangle)
    '''

    E, nu, t = Ge
    D = element.DLST(E, nu)  # Material matrix
    k = np.zeros((12, 12))   # Initialize stiffness matrix

    # Gauss integration weights and barycentric coordinates
    weights = [-27/48 , 25/48 , 25/48 , 25/48]
    for i,z in enumerate([[1/3,1/3,1/3],[3/5,1/5,1/5],[1/5,3/5,1/5],[1/5,1/5,3/5]],start=0):
        B, J = BLST_iso(Xe, z)  # Compute strain-displacement matrix and Jacobian
        detJ = np.linalg.det(J)
        if detJ <= 0:  # Checking that node configuration / element is physically possible
            raise ValueError(f"Jacobian determinant non-positive: detJ={detJ} at integration point {i}")
        k += 0.5 * weights[i] * (B.T @ D @ B) * detJ

    return k * t  # Multiply by element thickness

def shape_funcs_LST_iso(z): 
    '''
    Returns the interpolation matrix
        z: array with barycentric coordinates; [z1, z2, z3]
    '''
    
    # shape functions
    N1 = 2 * z[0] * (z[0] - 0.5)
    N2 = 2 * z[1] * (z[1] - 0.5)
    N3 = (2*z[0] + 2*z[1] - 2)*(z[0] + z[1] - 0.5)
    N4 = 4*z[1] * (1 - z[0] - z[1])
    N5 = 4*z[0] * (1 - z[0] - z[1])
    N6 = 4*z[0]*z[1]

    return np.array([N1, N2, N3, N4, N5, N6])




def buildR(X, T, Df, pL, bL, dL): 
    '''
    Return system load vector
        X: Coordinate matrix
        T: Topology matrix
        D: dof matrix
        
        pL: Point load
        bL: Boundary load  
        dL: Domain load

        nel: number of elements
        nno: number of nodes
    '''
    # init system load vector
    RL=np.zeros(np.max(Df)) 
    
    # point load
    if pL.any():
        # dofs with load
        de = pL[:,0].astype('i4')
        dei = de-1 # dof index
        # add nodal forcess
        RL[dei] += pL[:,1]  
    
    # boundary load
    if bL.any():

        # 3-point Gauss quadrature on [0,1]
        gauss_points = [0.1127, 0.5, 0.8873]
        gauss_weights = [5/18, 8/18, 5/18]

        # Local edge node indices (1-based → corrected to 0-based later)
        edge_nodes = {
            1: [1, 2, 6],
            2: [2, 3, 4],
            3: [3, 1, 5],
        }

        for i in range(len(bL)):
            el = int(bL[i, 0]) - 1
            si = int(bL[i, 1])
            qx_start, qx_end = bL[i, 2], bL[i, 3]
            qy_start, qy_end = bL[i, 4], bL[i, 5]

            node_ids = T[el, :] - 1   # shape (6,)
            Xe = X[node_ids]          # shape (6, 2)
            dofs_idx = Df[el] - 1     # shape (12,)

            edge_idx = [j - 1 for j in edge_nodes[si]]  # 0-based local indices
            Xe_edge = Xe[edge_idx]

            re = np.zeros(12)

            for gp, w in zip(gauss_points, gauss_weights):
                if si == 1:
                    z = [gp, 1 - gp, 0]
                elif si == 2:
                    z = [0, gp, 1 - gp]
                else:
                    z = [1 - gp, 0, gp]
                z = np.array(z)

                # shape functions
                z1, z2, z3 = z
                N1 = 2*z1*(z1-0.5)
                N2 = 2*z2*(z2-0.5)
                N3 = 2*z3*(z3-0.5)
                N4 = 4*z2*z3
                N5 = 4*z1*z3
                N6 = 4*z1*z2
                N = np.array([N1, N2, N3, N4, N5, N6])

                # Build N matrix (2x12)
                Nmat = np.zeros((2, 12))
                for j in range(6):
                    Nmat[0, 2*j]   = N[j]
                    Nmat[1, 2*j+1] = N[j]

                # Linearly interpolate traction
                qx = (1 - gp) * qx_start + gp * qx_end
                qy = (1 - gp) * qy_start + gp * qy_end
                t_vec = np.array([qx, qy])

                dx_dt = Xe_edge[1] - Xe_edge[0]
                J = np.linalg.norm(dx_dt) / 2

                re += (Nmat.T @ t_vec) * J * w

            RL[dofs_idx] += re
        
    # domain load #! BEDRE NUMERISK INTEGRATION!
    if dL.any(): 
        for i in range(len(dL)):
            # element
            el = dL[i,0].astype('i4')
            eli = el-1 # index
            # Nodes
            no1 = T[eli,0] # first
            no2 = T[eli,1] # second
            no3 = T[eli,2] # third
            # index
            no1i = no1-1
            no2i = no2-1
            no3i = no3-1
            # coordinates
            X1 = X[no1i,:]
            X2 = X[no2i,:]
            X3 = X[no3i,:]
            # element load [x,y] direction
            dLe = dL[i,[1,2]]

            qx = dL[i,1]
            qy = dL[i,2]

            # area of element
            A = 1/2*abs(X1[0]*(X2[1]-X3[1])+
                        X2[0]*(X3[1]-X1[1])+X3[0]*(X1[1]-X2[1]))
            # areal coordinates for centre
            z = [1/3,1/3,1/3]
            # interpolation matrix
            N = NLST(z)
            # calculate contribution from domain load
            rd = A * (N.T @ dLe)
            # array for dofs
            de = Df[eli,:] 
            dei = de-1 # dof index
            # add to system load vector
            RL[dei] += rd  
            
    return RL