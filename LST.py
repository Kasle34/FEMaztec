import numpy as np

def buildDf(T,nel): 
    '''
    Returns the system dof matrix
        T: Topology matrix
        nel: number of elmeents
    
    '''
    Df = np.zeros((nel,12)).astype('i4') # Defining the inital dof matrix 
    for el in range(1,nel+1): # for all elements 
         eli = el-1 # element index 
         # Nodes from Topology matrix
         no1 = T[eli,0] # first    
         no2 = T[eli,1] # second    
         no3 = T[eli,2] # third
         no4 = T[eli,3] # fourth (1st middle node)     
         no5 = T[eli,4] # fifth    
         no6 = T[eli,5] # six
         # previous dofs
         pdof1 = 2*(no1-1) # node 1
         pdof2 = 2*(no2-1) # node 2
         pdof3 = 2*(no3-1) # node 3
         pdof4 = 2*(no4-1) # node 4
         pdof5 = 2*(no5-1) # node 5
         pdof6 = 2*(no6-1) # node 6
         # dofs
         Df[eli,[0,1]] = pdof1+[1,2] # node 1
         Df[eli,[2,3]] = pdof2+[1,2] # node 2 
         Df[eli,[4,5]] = pdof3+[1,2] # node 3
         Df[eli,[6,7]] = pdof4+[1,2] # node 4
         Df[eli,[8,9]] = pdof5+[1,2] # node 5
         Df[eli,[10,11]] = pdof6+[1,2] # node 6
    return Df

def buildK(X, T, G, Df, nel):
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
        no1 = T[eli,0] # first 
        no2 = T[eli,1] # second
        no3 = T[eli,2] # third
        # index (adjusting for 1-based indexing in coordinate matrix)
        no1i = no1-1 # node 1 
        no2i = no2-1 # node 2
        no3i = no3-1 # node 3
        # element stiffness matrix 
        k = kLST(X[no1i],X[no2i],X[no3i],G[eli]) 
        # array for dofs
        de = Df[eli,:] # dofs
        dei = de-1 # dof index
        # assembling K
        K[np.ix_(dei,dei)] += k
    return K

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
        for i in range(len(bL)):
            # element
            el = bL[i,0].astype('i4')
            eli = el-1 # index
            # side
            si = bL[i,1].astype('i4')
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
            # boundary loads
            qx_start, qx_end = bL[i,2] , bL[i,3]
            qy_start, qy_end = bL[i,4] , bL[i,5]

            if si == 1: # if the first node
                # length of side
                L = np.sqrt((X2[0]-X3[0])**2+(X2[1]-X3[1])**2)
                # distributing boundary load to nodes. See (8.25) BK4 for example
                rb = np.array([0 , 0 , L*(qx_start-qx_end)/6+(L*qx_end)/6, L*(qy_start-qy_end)/6+(L*qy_end)/6 , L*qx_end/6 , L*qy_end/6 , L*(qx_start-qx_end)/3+2*L*qx_end/3 , L*(qy_start-qy_end)/3+2*L*qy_end/3 , 0 , 0 , 0 , 0])
            elif si == 2: # if the second node
                # length of side
                L = np.sqrt((X1[0]-X3[0])**2+(X1[1]-X3[1])**2)
                # distributing boundary load to nodes
                rb = np.array([ L*(-qx_start+qx_end)/6+(L*qx_start)/6, L*(-qy_start+qy_end)/6+(L*qy_start)/6 , 0, 0, L*qx_start/6 , L*qy_start/6, 0, 0, L*(-qx_start+qx_end)/3+2*L*qx_start/3, L*(-qy_start+qy_end)/3+2*L*qy_start/3, 0, 0])
            else: # if third node
                # length of side
                L = np.sqrt((X1[0]-X2[0])**2+(X1[1]-X2[1])**2)
                # distributing boundary load to nodes.
                rb = np.array([L*qx_start/6 , L*qy_start/6 , L*(-qx_start+qx_end)/6+(L*qx_start)/6 , L*(-qy_start+qy_end)/6+(L*qy_start)/6 , 0, 0, 0, 0, 0, 0, L*(-qx_start+qx_end)/3+2*L*qx_start/3 , L*(-qy_start+qy_end)/3+2*L*qy_start/3])
            # array for dofs
            de = Df[eli,:] 
            dei = de-1 # dof index
            # add to system load vector
            RL[dei] += rb
        
    # domain load 
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

            # area of element
            A = 1/2*abs(X1[0]*(X2[1]-X3[1])+
                        X2[0]*(X3[1]-X1[1])+X3[0]*(X1[1]-X2[1]))
            # area coordinates for centre
            z = [1/3,1/3,1/3]
            # interpolation matrix
            N = NLST(z)
            # calculate contri$bution from domain load
            rd = A * (N.T @ dLe)
            # array for dofs
            de = Df[eli,:] 
            dei = de-1 # dof index
            # add to system load vector
            RL[dei] += rd  
            
    return RL

def kLST(X1, X2, X3, Ge): 
    '''
    Returns element stiffness matrix
        X1, X2, X3: coordinates to element CORNER nodes
        Ge: element material properties'
    '''
    
    E, nu, t = Ge                                                               # Extracting material properties
    A = 0.5*abs(X1[0]*(X2[1]-X3[1])+X2[0]*(X3[1]-X1[1])+X3[0]*(X1[1]-X2[1]))    # Area of element
    D = DLST(E, nu)                                                             # Material matrix 
    k = np.zeros((12, 12))                                                      # Initialize stiffness matrix

    weights = [-27/48 , 25/48 , 25/48 , 25/48]
    for i,z in enumerate([[1/3,1/3,1/3],[3/5,1/5,1/5],[1/5,3/5,1/5],[1/5,1/5,3/5]],start=0):
        B =BLST(X1,X2,X3,z)
        k += weights[i] * (B.T @ D @ B)
    k *= A*t # Multiplying with thickness and (all sums) area

    return k

def NLST(z): 
    '''
    Returns the interpolation matrix
        z: array with barycentric coordinates; [z1, z2, z3]
    '''
    z1 ,z2 , z3= z[0], z[1], z[2]

    # shape functions
    N1 = 2*z1*(z1-0.5)
    N2 = 2*z2*(z2-0.5)
    N3 = 2*z3*(z3-0.5)
    N4 = 4*z2*z3
    N5 = 4*z1*z3
    N6 = 4*z1*z2

    # interpolation matrix
    N = np.array([[N1, 0, N2, 0, N3, 0, N4, 0, N5, 0, N6, 0],
                  [0, N1, 0, N2, 0, N3, 0, N4, 0, N5, 0, N6]])

    return N

def BLST(X1,X2,X3,z):
    ''' 
    Returns the strain interpolation matrix
        X1, X2 and X3: coordinates to element CORNER nodes
        z: Array with zeta-values for a point in the triangle, i.e. z = [0.15 , 0.45 , 0.4]
    '''
    
    z1 ,z2 , z3= z[0], z[1], z[2]
    a1, b1 = X3[0]-X2[0] , X3[1]-X2[1]
    a2, b2 = X1[0]-X3[0] , X1[1]-X3[1]
    a3, b3 = X2[0]-X1[0] , X2[1]-X1[1]

    # area of element
    A = 0.5*abs(X1[0]*(X2[1]-X3[1]) + X2[0]*(X3[1]-X1[1]) + X3[0]*(X1[1]-X2[1]))

    #Below functions are excluding 1/(2*A) - that is added when creating the B-matrix
    dN1dx = (4*z1-1)*(-b1)
    dN1dy = (4*z1-1)*a1

    dN2dx = (4*z2-1)*(-b2) 
    dN2dy = (4*z2-1)*a2 

    dN3dx = (4*z3-1)*(-b3) 
    dN3dy = (4*z3-1)*a3 

    dN4dx = 4*z3*(-b2) + 4*z2*(-b3)
    dN4dy = 4*z3*(a2) + 4*z2*(a3)
    
    dN5dx = 4*z3*(-b1) + 4*z1*(-b3)
    dN5dy = 4*z3*(a1) + 4*z1*(a3)

    dN6dx = 4*z2*(-b1) + 4*z1*(-b2)
    dN6dy = 4*z2*a1 + 4*z1*a2

    # strain interpolation matrix
    B = 1/(2*A)*np.array([
        (dN1dx , 0 , dN2dx , 0 , dN3dx , 0 , dN4dx , 0 , dN5dx , 0 , dN6dx , 0),
        (0 , dN1dy , 0 , dN2dy , 0 , dN3dy , 0 , dN4dy , 0 , dN5dy , 0 , dN6dy),
        (dN1dy , dN1dx , dN2dy , dN2dx , dN3dy , dN3dx , dN4dy , dN4dx , dN5dy , dN5dx , dN6dy , dN6dx),
        ])
    return B

def DLST(E,nu): 
    '''
    Returns the material matrix
        E: Young's modulus
        nu: Poisson's ratio
    '''
    # plane stress
    D = E/(1-nu**2) * np.array([
                                ( 1, nu,          0),
                                (nu,  1,          0),
                                ( 0,  0, 1/2*(1-nu)),
                                ])
    # # plane strain
    # D = E/((1+nu)*(1-2*nu)) * np.array([
    #                                     (1-nu,   nu,       0),
    #                                     (  nu, 1-nu,       0),
    #                                     (   0,    0, 1/2*-nu),
    #                                     ])
    return D

def Eps(X1,X2,X3,Ve,z): 
    '''
    Returns FEM generalized element strain
        X1 and X2: coordinates to element end nodes
        Ge: element material properties
        Ve: element displacement vector
        z: Barycentric coordinates of point of stress-calculation
    '''
    B = BLST(X1,X2,X3,z) # strain interpolation matrix
    e = (B @ Ve) # generelised strains
  
    return e

def S(X1,X2,X3,Ge,Ve,z): 
    '''
    Returns FEM generalized element stress
        X1 and X2: coordinates to element end nodes
        Ge: element material properties
        Ve: element displacement vector
        z: Barycentric coordinates of point of stress-calculation
    '''
    D = DLST(Ge[0],Ge[1]) # material stiffness matrix
    B = BLST(X1,X2,X3,z) # strain interpolation matrix
    s = (D @ B @ Ve) # generelised stresses
    
    return s

def S_Eps(X1,X2,X3,Ge,Ve0,Ve1,z):
    '''
    Returns FEM generalized element stress & strains (combination of S & Eps for faster computation)
        X1 and X2: coordinates to element end nodes
        Ge: element material properties
        Ve: element displacement vector
        z: Barycentric coordinates of point of stress-calculation
    '''
    D = DLST(Ge[0],Ge[1]) # material stiffness matrix
    B = BLST(X1,X2,X3,z) # strain interpolation matrix

    s = (D @ B @ Ve0) # generelised stresses
    e = (B @ Ve1) # generelised strains
    
    return s, e 