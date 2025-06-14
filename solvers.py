import numpy as np

def linearLST(K,RL,U):
    '''
    Returns displacements at non-prescribed dofs, V
    Returns reactions at prescribed dofs, Ru
        K = global stiffness matrix
        RL = loads: array("node", "dof/dir", "force")
        U = displacement loads: array("node", "dof/dir", "value")
    '''
    ndof = len(K) # determines the number of dofs in the system
    da=np.arange(1,ndof+1) # all dofs

    du = U[:,0].astype('i4') # dofs with prescribed displacements 
    Vu = U[:,1] # prescribed displacements 

    df=np.setdiff1d(da,du) # all free dofs 
    dfi = df-1 
    dui = du-1 
    Rf = RL[dfi]
    Kff = K[np.ix_(dfi,dfi)]
    Kfu = K[np.ix_(dfi,dui)]
    Kuf = K[np.ix_(dui,dfi)]
    Kuu = K[np.ix_(dui,dui)]

    # solve equation system
    A = Kff
    B = Rf-Kfu@Vu
    
    Vf=np.linalg.solve(A,B) # solve for the displacements (dofs)
    Ru=Kuf@Vf+Kuu@Vu
    
    V = np.zeros(ndof) # re-assemble dof vector
    V[dfi]=Vf # free dofs
    V[dui]=Vu # supported dofs
    
    R = np.zeros(ndof) # re-assemble nodal force vector
    R[dfi] = Rf # nodal force for free dofs 
    R[dui] = Ru # nodal force for supported dofs

    return V, Ru

