import numpy as np

def generate_LST_mesh(length, height, nelx, nely):
    """
    Generates an LST mesh with user-defined number of elements in width (nelx) and height (nely).
    
    Parameters:
        length (float): Length of the domain in the x-direction.
        height (float): Height of the domain in the y-direction.
        nelx (int): Number of elements in the x-direction.
        nely (int): Number of elements in the y-direction.
        
    Returns:
        X (np.ndarray): Node coordinates.
        T (list): List of LST element connectivity (6 nodes per element).
    """
    nnox = 2 * nelx + 1  # Number of nodes in x-direction
    nnoy = 2 * nely + 1  # Number of nodes in y-direction
    
    # Generate node coordinates
    X = np.zeros((nnox * nnoy, 2))
    no = 0
    for j in range(nnoy):
        for i in range(nnox):
            X[no, :] = [length * i / (nnox - 1), height * j / (nnoy - 1)]
            no += 1
    
    # Generate element topology
    T = []
    el_ind = 0
    for i in range(nely):
        for j in range(nelx):
            base_index = (i * 2) * nnox + (j * 2)
            
            # Corner nodes
            c1 = base_index
            c2 = base_index + 2
            c3 = base_index + 2 * nnox + 2
            c4 = base_index + 2 * nnox
            
            # Mid-line nodes
            ml1 = base_index + 1
            ml2 = base_index + nnox + 2
            ml3 = base_index + 2 * nnox + 1
            ml4 = base_index + nnox
            
            # Center node
            center = base_index + nnox + 1
            
            # Add two LST elements
            T.append(np.array([c1, c3, c4, ml3, ml4, center]))
            T.append(np.array([c1, c2, c3, ml2, center, ml1]))
            
            el_ind += 2
    
    return np.array(X), np.array(T)+1
