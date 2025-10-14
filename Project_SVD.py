import numpy as np

def SVD(A):
    """
    Applies a SVD decomposition to a nxm matrix A, returning all of the matrices of SVD, the condition number, and the inverse of A if it exists.

    Paramaters:
    A (np.ndarray): The nxm matrix where SVD will be applied

    Return:
    U (np.ndarray): The U matrix of SVD decomposition 
    Sigma (np.ndarray): The Sigma matrix of SVD decomposition (singular values)
    V.T (np.ndarray): The V transpose matrix of SVD decomposition 
    cond_number (float): The condition number of the matrix A
    A_inv (np.ndarray) or Error("string"): The inverse of the input matrix if it exists if not return error
    singular, sort_eigval (np.ndarray): The sorted eigen values and the singular values of the matrix A
    """
    # Getting shape
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix is rectangular, inverse does not exist")
    
    ATA = np.dot(A.T, A)

    # Now we find the eigen values and vectors of ATA and sort in Descending order 
    eigval, eigvec = np.linalg.eigh(ATA)

    # Keep index for ordering
    sorted_index = np.argsort(eigval)[::-1]

    # Apply index
    sort_eigvec = eigvec[:, sorted_index]
    sort_eigval = eigval[sorted_index]

    # Singular values are the square roots of all the eigen values
    singular = np.sqrt(np.clip(sort_eigval, 0, None))
    # Build the sigma matrix as a rectangular diagonal matrix which is the original shape of the input
    Sigma = np.zeros((n, m))
    np.fill_diagonal(Sigma, singular)

    # Build U 
    U = np.zeros((n,n))
    V = sort_eigvec

    # If there exists a 0 singular value then continue the SVD but use 0 as the orthonormal vector => Not Invertible 
    # ui​=(1/σi)​​A(vi​)
    for i in range(len(singular)):
        if singular[i] > 1e-6:
            U[:, i] = np.dot(A, V[:, i]) / singular[i]
        else:
            raise ValueError("Error: A singular value is zero, inverse does not exist")

    # Condition number using the 2-norm = σ_max(A) / σ_min(A)
    # Make sure we don't divide by zero
    nonzero_singular = singular[singular > 1e-10]
    if len(nonzero_singular) == 0:
        raise ValueError("All singular values are zero.")
    cond_number = np.max(singular) / np.min(nonzero_singular)
    
    # Computing matrix inverse
    if np.min(singular) < 1e-6:
        raise ValueError("Error: matrix is singular, inverse does not exist")
    
    S_inv = np.zeros((n, m))
    np.fill_diagonal(S_inv, 1.0 / singular)
    # Use note in project
    A_inv = np.dot(V, np.dot(S_inv, U.T))

    return U, Sigma, V.T, cond_number, A_inv, singular, sort_eigval

# Part 2 Spring System
def build_A(nmasses, nspring, boundary):
    """
    Builds the A matrix for the system for the case of two fixed ends (Both) and one fixed end (Single)

    Parameters:
    nmasses(float): The number of masses of the system
    nspring(float): The number of springs in the system
    boundary(string): The boundary condition to applt to the problem
    
    Returns:
    A (np.ndarray): The A matrix for the spring mass system
    """
    if boundary == "Both":
        if nspring != nmasses + 1:
            raise ValueError(f"Warning: For 'Both' boundary, typically nspring = nmasses + 1")

        A = np.zeros((nspring, nmasses))
        for i in range(0, nspring - 1):
            A[i, i] = 1
            A[i + 1, i] = -1

    elif boundary == "Single":
        if nspring != nmasses:
            raise ValueError(f"Warning: For 'Single' boundary, typically nspring = nmasses")

        A = np.zeros((nmasses, nmasses))
        for i in range(0, nmasses - 1):
            A[i,i] = 1
            A[i + 1 , i] = -1
        A[-1, -1] = 1 
    return A

def build_C(kconstant):
    """
    Builds the C matrix containing all of the spring stiffnesses of the system

    Paramters:
    kconstant (list): A list of all the spring constants in order from the top to the bottom of the system

    Returns:
    C (np.ndarray): The C matrix where all spring constants are its diagonals
    """
    n = len(kconstant)
    C = np.zeros((n,n))
    np.fill_diagonal(C, kconstant)

    return C

def spring_mass(nmasses, nspring, kconstant, boundary, masses):
    """
    Solves a spring mass system for displacement, elongation, and stress for two cases. (Two fixed ends, one fixed ends)
    Prints out the singular and eigen values of the stiffness matrix through the SVD function

    Paramaters:
    nmasses(float): The number of masses of the system
    nspring(float): The number of springs in the system
    kconstant(list): A list of all the spring constants in order from the top to the bottom of the system
    boundary(string): The boundary condition to apply to the problem ('Both' = fixed-fixed) ('Single' = fixed-free)
    masses(list): A list of all the mass values in order from the top mass to the bottom mass of the system
    
    Returns:
    displacements(np.ndarray)
    elongations(np.ndarray)
    stress(np.ndarray)
    Condtition(float)
    K(np.ndarray)
    """
    A = build_A(nmasses, nspring, boundary)
    C = build_C(kconstant)

    F = np.zeros([nmasses, 1])
    for i in range(len(masses)):
        F[i, 0] = masses[i] * 9.81

    # solve for u Ku=f using SVD
    K = A.T @ C @ A
    
    U, S, VT, Condition, K_inverse, singular, sort_eigval = SVD(K)
    # Print singular values and eigen values
    print("Singular values of K:")
    print(singular)

    print("Eigen values of K:")
    print(sort_eigval)
    displacements = K_inverse @ F

    elongations= np.dot(A, displacements)
    stress = np.dot(C, elongations)

    return displacements, elongations, stress, Condition, K


# Compare Results Examination for Invertible matrix with Black Box
A = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
U, Sigma, VT, cond_number, A_inv, singular, sort_eigval = SVD(A)

U_bb, S_BB, VT_bb = np.linalg.svd(A)
print("U Comparison")
print("My U:\n", U)
print("NumPy U:\n", U_bb)
print("\nSigma")
print("My Sigma:\n", Sigma)
print("NumPy Sigma (1D):\n", S_BB)
print("\nVT Comparison")
print("My VT:\n", VT)
print("NumPy VT:\n", VT_bb)
print("\nCondition Number")
print("My cond number:", cond_number)
print("NumPy cond number:", np.linalg.cond(A))
print("\nInverse Comparison\n")
print("My Inverse:\n", A_inv)
print("NumPy Inverse:\n", np.linalg.inv(A))

