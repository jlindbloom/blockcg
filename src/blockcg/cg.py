import numpy as np
from concurrent.futures import ProcessPoolExecutor


def cg(A, b, x0=None, M=None, tol=1e-8, maxiter=None, xtrue=None, K=None):
    """Standard conjugate gradient method.
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

    # Define matvec if A is a matrix
    r = b - A @ x
    if M is not None:
        z = M @ r
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
        else:
            z = r
        if K is not None: z -= K @ (K.T @ z) # remove part from the kernel
        rz_new = np.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        if K is not None: p -= K @ (K.T @ p) # remove part from the kernel
        rz_old = rz_new

    info = {
        'residual_norms': residual_norms,
        'rel_residual_norms': rel_residual_norms,
        'num_iters': k + 1,
        'success': success
    }

    if K is None:
        info["K_proj_norm"] = 0.0
    else:
        info["K_proj_norm"] = np.linalg.norm( K @ ( K.T @ x ) )

    if track_error:
        info['abs_A_errors'] = abs_A_errors
        info['rel_A_errors'] = rel_A_errors
        info["abs_two_errors"] = abs_two_errors
        info["rel_two_errors"] = rel_two_errors
        
    return x, info





def parallelcg(A, B_rhs, X0=None, tol=1e-3, M=None, Xtrue=None, maxiter=None, K=None):
    """Solves A X = B by running cg in parallel for each column.
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

    # Counter for iteration
    n_iters = 0

    # Initialization
    if X0 is None:
        X0 = np.zeros((n,s))
    else:
        pass
        
    inputs = [ ( 
        (A, B[:,j]), 
            {   "x0": X0[:,j],
                "tol": tol,
                "maxiter":maxiter,
                "xtrue": Xtrue[:,j],
                "K": K,
        }) for j in range(B.shape[1]) ]
    

    with ProcessPoolExecutor() as exe:
        results = list(exe.map(cg_wrapper, inputs))

    # Reform X
    X = np.zeros((n,s))
    infos = []
    for j in range(B.shape[1]):
        X[:,j] = results[j][0]
        infos.append( results[j][1] )
    
    return X, infos




def cg_wrapper(args_kwargs):
    args, kwargs = args_kwargs
    return cg(*args, **kwargs)