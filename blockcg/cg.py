import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time


def cg(A, b, x0=None, M=None, tol=1e-8, maxiter=None, xtrue=None, K=None):
    """
    Solve the linear system Ax = b using the Conjugate Gradient (CG) method.
    Parameters
    ----------
    A : array_like or linear operator
        Symmetric positive-definite matrix or linear operator representing the system.
    b : array_like
        Right-hand side vector.
    x0 : array_like, optional
        Initial guess for the solution. If None, zeros are used.
    M : array_like or linear operator, optional
        Preconditioner matrix or operator. If None, no preconditioning is applied.
    tol : float, optional
        Tolerance for convergence based on the relative residual norm. Default is 1e-8.
    maxiter : int, optional
        Maximum number of iterations. If None, defaults to the size of b.
    xtrue : array_like, optional
        True solution vector, used for error tracking. If provided, errors are computed at each iteration.
    K : array_like, optional
        Matrix whose columns span the kernel to be projected out at each iteration.
    Returns
    -------
    x : ndarray
        Approximate solution to Ax = b.
    info : dict
        Dictionary containing convergence information and statistics:
            - 'residual_norms': array of residual norms at each iteration.
            - 'rel_residual_norms': array of relative residual norms at each iteration.
            - 'num_iters': number of iterations performed.
            - 'converged': boolean indicating if convergence was achieved.
            - 'tot_time': total elapsed time for the algorithm.
            - 'tot_matvec_A': total number of matrix-vector products with A.
            - 'tot_matvec_M': total number of matrix-vector products with M.
            - 'K_proj_norm': norm of the projection onto the kernel (if K is provided).
            - 'abs_A_errors', 'rel_A_errors': absolute and relative errors in A-norm (if xtrue is provided).
            - 'abs_two_errors', 'rel_two_errors': absolute and relative errors in 2-norm (if xtrue is provided).
    Notes
    -----
    This implementation supports optional preconditioning and kernel projection.
    Error tracking is available if the true solution is provided.
    """
    b = np.asarray(b)
    n = b.shape[0]
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = np.array(x0, copy=True)
        if K is not None: x -= K @ (K.T @ x) # remove part from the kernel
    
    if maxiter is None:
        maxiter = n

    tot_start_time = time.perf_counter()
    tot_matvec_A = 0
    tot_matvec_M = 0

    # Define matvec if A is a matrix
    r = b - A @ x
    tot_matvec_A += 1
    if M is not None:
        z = M @ r
        tot_matvec_M += 1
    else:
        z = r
    if K is not None: z -= K @ (K.T @ z) # remove part from the kernel
    p = z.copy()
    rz_old = np.dot(r, z)

    norm_b = np.linalg.norm(b)
    norm_r = np.linalg.norm(r)
    rel_residual = norm_r if norm_b == 0 else norm_r / norm_b
    residual_norms = [norm_r]
    rel_residual_norms = [rel_residual]
    success = False

    # Error tracking
    track_error = xtrue is not None
    if track_error:
        xtrue = np.asarray(xtrue)
        # Precompute denom for relative error
        Axtrue = A @ xtrue
        true_A_norm = np.sqrt(np.dot(xtrue, Axtrue))
        true_two_norm = np.sqrt(np.dot(xtrue, xtrue))
        abs_A_errors = []
        rel_A_errors = []
        abs_two_errors = []
        rel_two_errors = []

        def A_norm_error(x):
            diff = x - xtrue
            return np.sqrt(np.dot(diff, A @ diff))
        
        def two_norm_error(x):
            diff = x - xtrue
            return np.sqrt(np.dot(diff, diff))

        abs_err = A_norm_error(x)
        rel_err = abs_err / true_A_norm if true_A_norm > 0 else np.nan
        abs_A_errors.append(abs_err)
        rel_A_errors.append(rel_err)
        abs_err = two_norm_error(x)
        rel_err = abs_err / true_two_norm if true_two_norm > 0 else np.nan
        abs_two_errors.append(abs_err)
        rel_two_errors.append(rel_err)

    for k in range(maxiter):
        Ap = A @ p
        tot_matvec_A += 1
        alpha = rz_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        if K is not None: 
            x -= K @ (K.T @ x) # remove part from the kernel
            r -= K @ (K.T @ r) 

        norm_r = np.linalg.norm(r)
        residual_norms.append(norm_r)

        if norm_b == 0:
            rel_resid = norm_r
        else:
            rel_resid = norm_r / norm_b
        rel_residual_norms.append(rel_resid)

        if track_error:
            abs_err = A_norm_error(x)
            rel_err = abs_err / true_A_norm if true_A_norm > 0 else np.nan
            abs_A_errors.append(abs_err)
            rel_A_errors.append(rel_err)
            abs_err = two_norm_error(x)
            rel_err = abs_err / true_two_norm if true_two_norm > 0 else np.nan
            abs_two_errors.append(abs_err)
            rel_two_errors.append(rel_err)

        if rel_resid < tol:
            success = True
            break

        if M is not None:
            z = M @ r
            tot_matvec_M += 1
        else:
            z = r
        if K is not None: z -= K @ (K.T @ z) # remove part from the kernel
        rz_new = np.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        if K is not None: p -= K @ (K.T @ p) # remove part from the kernel
        rz_old = rz_new

    tot_end_time = time.perf_counter()
    tot_time = tot_end_time - tot_start_time

    info = {
        'residual_norms': np.asarray(residual_norms),
        'rel_residual_norms': np.asarray(rel_residual_norms),
        'num_iters': k + 1,
        'converged': success,
        "tot_time": tot_time,
        "tot_matvec_A": tot_matvec_A,
        "tot_matvec_M": tot_matvec_M,
    }

    if K is None:
        info["K_proj_norm"] = 0.0
    else:
        info["K_proj_norm"] = np.linalg.norm( K @ ( K.T @ x ) )

    if track_error:
        info['abs_A_errors'] = np.asarray(abs_A_errors)
        info['rel_A_errors'] = np.asarray(rel_A_errors)
        info["abs_two_errors"] = np.asarray(abs_two_errors)
        info["rel_two_errors"] = np.asarray(rel_two_errors)
        
    return x, info





def parallelcg(A, B_rhs, X0=None, tol=1e-3, M=None, Xtrue=None, maxiter=None, K=None):
    """
    Solves multiple symmetric positive definite (SPD) linear systems in parallel using the Conjugate Gradient (CG) method.
    Each column of B_rhs is treated as a separate right-hand side, and the corresponding solution is computed in parallel.
    Optionally, preconditioning and true solutions for error analysis can be provided.
    Parameters
    ----------
    A : np.ndarray
        SPD matrix of shape (n, n).
    B_rhs : np.ndarray
        Right-hand side matrix of shape (n, s), where each column is a separate system.
    X0 : np.ndarray, optional
        Initial guess for the solution, shape (n, s). If None, zeros are used.
    tol : float, optional
        Tolerance for convergence. Default is 1e-3.
    M : np.ndarray or callable, optional
        Preconditioner for A. Default is None.
    Xtrue : np.ndarray, optional
        True solution matrix for error analysis, shape (n, s). Default is None.
    maxiter : int, optional
        Maximum number of iterations for CG. Default is n.
    K : any, optional
        Optional projection or additional parameter for CG. Default is None.
    Returns
    -------
    X : np.ndarray
        Solution matrix of shape (n, s), where each column is the solution to A x = b.
    info : dict
        Dictionary containing convergence information and statistics:
            - n_iters: List of iteration counts for each system.
            - converged: List of booleans indicating convergence for each system.
            - K_proj_norm: Sum of projection norms (if applicable).
            - tot_time: Total computation time.
            - tot_matvec_A: Total number of matrix-vector products with A.
            - tot_matvec_M: Total number of matrix-vector products with M.
            - all_converged: Boolean indicating if all systems converged.
            - elem_rel_residual_norms: Array of relative residual norms for each system.
            - elem_A_errs: Array of absolute A-norm errors (if Xtrue is provided).
            - elem_two_errs: Array of absolute 2-norm errors (if Xtrue is provided).
    """

    # Copy B
    B = B_rhs.copy()

    # Shapes
    assert A.shape[0] == A.shape[1], "A must be SPD!"
    s = B.shape[1]
    n = A.shape[0]
    assert B.shape[0] == n, "shape of B not compatible with A!"

    # Handle maxiter
    if maxiter is None:
        maxiter = n

    # Initialization
    if X0 is None:
        X0 = np.zeros((n,s))
    else:
        pass

    tot_start_time = time.perf_counter()
    

    if Xtrue is not None:
        
        inputs = [ ( 
            (A, B[:,j]), 
                {   "x0": X0[:,j],
                    "tol": tol,
                    "maxiter":maxiter,
                    "xtrue": Xtrue[:,j],
                    "K": K,
                    "M": M,
            }) for j in range(B.shape[1]) ]
    
    else:

        inputs = [ ( 
        (A, B[:,j]), 
            {   "x0": X0[:,j],
                "tol": tol,
                "maxiter":maxiter,
                "xtrue": None,
                "K": K,
                "M": M,
        }) for j in range(B.shape[1]) ]

    with ProcessPoolExecutor() as exe:
        results = list(exe.map(cg_wrapper, inputs))

    tot_end_time = time.perf_counter()
    tot_time = tot_start_time - tot_end_time
    # Reform X
    X = np.zeros((n,s))

    info = {
        "n_iters": [],
        "converged": [],
        "K_proj_norm": 0.0,
        "tot_time": tot_time,
        "tot_matvec_A": 0,
        "tot_matvec_M": 0,
        "all_converged": True,
        "elem_rel_residual_norms": [],
    }


    largest_len = 0
    for j in range(B.shape[1]):
        X[:,j] = results[j][0]

        sub_info = results[j][1]

        info["n_iters"].append( sub_info["num_iters"] )
        info["converged"].append( sub_info["converged"] )
        info["tot_time"] += sub_info["tot_time"]
        info["tot_matvec_A"] += sub_info["tot_matvec_A"]
        info["tot_matvec_M"] += sub_info["tot_matvec_M"]
        info["K_proj_norm"] += sub_info["K_proj_norm"]

        if len(sub_info["rel_residual_norms"]) > largest_len:
            largest_len = len(sub_info["rel_residual_norms"])

        # elem_rel_residual_norms.append( sub_info["rel_residual_norms"] )

        # if Xtrue is not None:
        #     elem_A_errs.append( sub_info["abs_A_errors"][:,None] )
        #     elem_two_errs.append( sub_info["abs_two_errors"][:,None] )
    
        if not sub_info["converged"]:
            info["all_converged"] = False

    # Pad to all have same length
    elem_rel_residual_norms = []
    elem_A_errs = []
    elem_two_errs = []
    for j in range(B.shape[1]):

        sub_info = results[j][1]
        if len(sub_info["rel_residual_norms"]) < largest_len:
            padded = np.pad( sub_info["rel_residual_norms"], (0, largest_len - len(sub_info["rel_residual_norms"])), mode='edge')
            sub_info["rel_residual_norms"] = padded[None,:]
        
            if Xtrue is not None:
                padded = np.pad( sub_info["abs_A_errors"], (0, largest_len - len(sub_info["abs_A_errors"])), mode='edge')
                sub_info["abs_A_errors"] = padded[None,:]

                padded = np.pad( sub_info["abs_two_errors"], (0, largest_len - len(sub_info["abs_two_errors"])), mode='edge')
                sub_info["abs_two_errors"] = padded[None,:]

        elem_rel_residual_norms.append(sub_info["rel_residual_norms"])
        if Xtrue is not None:
            elem_A_errs.append(sub_info["abs_A_errors"])
            elem_two_errs.append(sub_info["abs_two_errors"])


    #info["elem_rel_residual_norms"] = np.hstack( elem_rel_residual_norms )
    info["elem_rel_residual_norms"] = np.vstack( elem_rel_residual_norms )

    if Xtrue is not None:
        info["elem_A_errs"] = np.vstack( elem_A_errs )
        info["elem_two_errs"] = np.vstack( elem_two_errs )


    return X, info



def cg_wrapper(args_kwargs):
    args, kwargs = args_kwargs
    return cg(*args, **kwargs)