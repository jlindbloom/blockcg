import numpy as np
from scipy.linalg import qr, svd
import scipy.sparse as sps



def calc_conv_characteristic(A, X, Xtrue, norm="A"):
    """Computes the convergence characteristic

    (  trace( ( X_true - X )^T A ( X_true - X ) ) / trace( X_true^T A X_true )   )^1/2.

    Returns the numerator, entire characteristic, and the numerator split up by its components.
    """
    assert (norm == "A") or (norm == "two"), "invalid norm!"

    if norm == "A":
        diff = Xtrue - X
        tmp = diff.T @ (A @ diff)
        num = np.trace( tmp )
        diag = np.diag( tmp )
        denom = np.trace( Xtrue.T @ ( A @ Xtrue)  )
    else:
        diff = Xtrue - X
        tmp = diff.T @ diff
        num = np.trace( tmp )
        diag = np.diag( tmp )
        denom = np.trace( Xtrue.T @ Xtrue )
        
    return np.sqrt(num), np.sqrt( num ) / np.sqrt( denom ), np.sqrt(diag)





def chan_rrqr(A, tau=1e-12, return_PiW=False):
    """Implements Chan's RRQR. 
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q, R = qr(A, mode='economic')
    perm = np.arange(n)
    W = np.eye(n)
    rank = 0

    for k in range(n, 0, -1):
        R11 = R[:k, :k]
        sigma_min, wk = smallest_singular(R11)
        if sigma_min > tau:
            rank = k
            break
        idx_max = np.argmax(np.abs(wk))
        if idx_max != k-1:
            R[:k, [k-1, idx_max]] = R[:k, [idx_max, k-1]]
            perm[[k-1, idx_max]] = perm[[idx_max, k-1]]
            W[[k-1, idx_max], :] = W[[idx_max, k-1], :]
            wk[[k-1, idx_max]] = wk[[idx_max, k-1]]
        Qtilde, Rtilde = qr(R[:k, :k], mode='economic')
        R[:k, :k] = Rtilde
        if k < n:
            R[:k, k:] = Qtilde.T @ R[:k, k:]
        if k < m:
            R[k:, :k] = 0
        Q[:, :k] = Q[:, :k] @ Qtilde
        W[:k, k-1] = wk
    if return_PiW:
        PiW = W[perm, :]
        return Q, R, perm, rank, W, PiW
    else:
        return Q, R, perm, rank, W


def smallest_singular(M):
    U, s, Vh = svd(M, full_matrices=False)
    idx = np.argmin(s)
    return s[idx], Vh[-1, :]





def all_residuals_converged(R, B, tol=1e-5, B_norms=None):
    """
    Check if all columns of residual R satisfy ||r_i|| < tol * ||b_i||.

    Parameters
    ----------
    R : ndarray, shape (n, m)
        Residual matrix (columns are residuals for each right-hand side).
    B : ndarray, shape (n, m)
        Right-hand side matrix (columns are b_i).
    tol : float
        Relative tolerance.

    Returns
    -------
    converged : bool
        True if all columns satisfy the criterion, False otherwise.
    """
    # Compute the 2-norm of each residual and rhs
    residual_norms = np.linalg.norm(R, axis=0)
    if B_norms is None:
        rhs_norms = np.linalg.norm(B, axis=0)
    else:
        rhs_norms = B_norms
    
    # For columns with zero RHS, use absolute residual
    test = np.empty_like(residual_norms, dtype=bool)
    for i in range(R.shape[1]):
        if rhs_norms[i] == 0:
            test[i] = residual_norms[i] < tol
        else:
            test[i] = residual_norms[i] < tol * rhs_norms[i]

    return np.all(test)




def first_order_derivative_1d(N, boundary="none"):
    """Constructs a sparse matrix that extracts the (1D) discrete gradient of an input signal.
    Boundary parameter specifies how to handle the boundary conditions. Also returns a dense matrix W
    whose column span the nullspace of the operator (if trivial, W = None).
    """
    
    assert boundary in ["none", "periodic", "zero", "reflexive", "zero_sym"], "Invalid boundary parameter."
    
    d_mat = sps.eye(N)
    d_mat.setdiag(-1,k=1)
    #d_mat = sps.csc_matrix(d_mat)
    d_mat = d_mat.tolil()
    
    if boundary == "periodic":
        d_mat[-1,0] = -1
        W = np.atleast_2d(np.ones(N)).T
    elif boundary == "zero":
        W = None
        pass
    elif boundary == "none":
        d_mat = d_mat[:-1,:]
        W = np.atleast_2d(np.ones(N)).T
    elif boundary == "reflexive":
        d_mat[-1,-1] = 0
        W = np.atleast_2d(np.ones(N)).T
    elif boundary == "zero_sym":
        d_mat = sps.csc_matrix(d_mat)
        new_row = sps.csc_matrix(np.zeros(d_mat.shape[1]))
        d_mat = sps.vstack([new_row, d_mat])
        d_mat[0,0] = -1
        W = None
    else:
        pass
    
    return d_mat, W



def laplacian2D_neumann(grid_shape):
    """Makes a sparse matrix for an discrete laplacian operator on a 2D grid.
    Uses Neumann boundary conditions (constant vector is in the kernel).

    Returns (A, K).
    """

    m, n = grid_shape

    Rv, _ = first_order_derivative_1d(m, boundary="reflexive")
    Rv *= -1.0

    Rh, _ = first_order_derivative_1d(n, boundary="reflexive")
    Rh *= -1.0

    L = sps.vstack([sps.kron(Rv, sps.eye(n)), sps.kron(sps.eye(m), Rh) ])
    A = L.T @ L

    K = np.ones(A.shape[1])
    K /= np.linalg.norm(K)
    K = K[:,None]

    return A, K



def laplacian2D_dirichlet(grid_shape):
    """Makes a sparse matrix for an discrete laplacian operator on a 2D grid.
    Uses zero Dirichlet boundary conditions.

    Returns (A, K).
    """

    m, n = grid_shape

    Rv, _ = first_order_derivative_1d(m, boundary="zero_sym")
    Rv *= -1.0

    Rh, _ = first_order_derivative_1d(n, boundary="zero_sym")
    Rh *= -1.0

    L = sps.vstack([sps.kron(Rv, sps.eye(n)), sps.kron(sps.eye(m), Rh) ])
    A = L.T @ L
    K = None

    return A, K






def random_spd_with_p_eigenvalues(N, p, eig_min=1.0, eig_max=10.0, rseed=None):
    """
    """
    rng = np.random.default_rng(rseed)

    # 1. Choose p distinct positive eigenvalues
    eigvals_distinct = np.linspace(eig_min, eig_max, p)

    # 2. Repeat each as needed to make N entries (with possible repeats)
    eigvals = np.tile(eigvals_distinct, N // p + 1)[:N]
    rng.shuffle(eigvals)  # Randomize their order
    
    # 3. Generate a random orthogonal matrix Q
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    
    # 4. Construct SPD matrix
    A = Q @ np.diag(eigvals) @ Q.T
    # Ensure symmetry (numerical)
    A = (A + A.T) / 2
    return A, eigvals