import numpy as np
from scipy.linalg import qr
import time
from blockcg.util import *




def blockcg(A, B_rhs, X0=None, tol=1e-5, M=None, Xtrue=None, variant="DP", maxiter=None, tau=1e-10, K=None):
    """
    Implements block Conjugate Gradient (block CG) methods for solving systems of linear equations with multiple right-hand sides.
    Supports the following block CG variants:
        - DP: Block CG with QR-based orthogonalization (Default)
        - BF: Block CG with Chan's RRQR-based orthogonalization
        - HS: Block CG with block-Hestenes-Stiefel update
    The method iteratively solves AX = B for X, where A is symmetric positive definite (SPD), and B may have multiple columns (right-hand sides).
    The stopping criterion is based on the relative residual norm for each column.
    Parameters
    ----------
    A : np.ndarray
        Symmetric positive definite matrix of shape (n, n).
    B_rhs : np.ndarray
        Right-hand side matrix of shape (n, s), where s is the number of right-hand sides.
    X0 : np.ndarray, optional
        Initial guess for the solution, shape (n, s). If None, zeros are used.
    tol : float, optional
        Tolerance for the relative residual stopping criterion (default: 1e-5).
    M : np.ndarray or callable, optional
        Preconditioner matrix or function. If None, no preconditioning is applied.
    Xtrue : np.ndarray, optional
        True solution for error analysis (used for convergence metrics).
    variant : str, optional
        Block CG variant to use: "DP", "BF", or "HS" (default: "DP").
    maxiter : int, optional
        Maximum number of iterations (default: n).
    tau : float, optional
        Tolerance for RRQR in the "BF" variant (default: 1e-10).
    K : np.ndarray, optional
        Matrix whose columns span the kernel to be projected out (deflation).
    Returns
    -------
    X : np.ndarray
        Approximate solution matrix of shape (n, s).
    info : dict
        Dictionary containing convergence information and metrics:
            - n_iters: Number of iterations performed
            - residual_norms: Frobenius norms of residuals at each iteration
            - converged: Boolean indicating convergence
            - block_sizes: Block sizes at each iteration
            - singular_breakdown: Boolean indicating if breakdown occurred
            - elem_rel_residual_norms: Relative residual norms per column
            - tot_time: Total runtime
            - timing_A: Time spent in matrix-vector products with A
            - timing_M: Time spent in preconditioning
            - tot_matvec_A: Total number of A matvecs
            - tot_matvec_M: Total number of M matvecs
            - K_proj_norm: Norm of the projection onto the kernel (if K is provided)
            - abs_conv_characteristics, rel_conv_characteristics: Convergence metrics (if Xtrue is provided)
            - abs_two_conv_characteristics, rel_two_conv_characteristics: Additional convergence metrics (if Xtrue is provided)
            - elem_A_errs, elem_two_errs: Elementwise error metrics (if Xtrue is provided)
    Notes
    -----
    - The function assumes A is symmetric positive definite (SPD).
    - Preconditioning is not supported for the "DR" variant.
    - If the block size deflates to zero or a singular breakdown occurs, the iteration terminates early.
    """
    
    # Valid variants
    valid_variants = ["DP", "BF", "HS"]
    assert variant in valid_variants, f"Invalid variant, must be one of {valid_variants}"
    if variant == "DR":
        assert M is None, "Preconditioning not currently supported if using DR variant."
    
    # Copy B
    B = B_rhs.copy()

    # Shapes
    assert A.shape[0] == A.shape[1], "A must be SPD!"
    s = B.shape[1]
    n = A.shape[0]
    assert B.shape[0] == n, "shape of B not compatible with A!"
    B_norms = np.linalg.norm(B, axis=0)

    # Handle maxiter
    if maxiter is None:
        maxiter = n

    # Counter for iteration
    n_iters = 0

    # Timing
    tot_start_time = time.perf_counter()
    timing_A = 0
    timing_M = 0

    # Matvec counting
    tot_matvec_A = 0
    tot_matvec_M = 0

    # Initial residual
    if X0 is None:
        Rcurr = B
        X = np.zeros((n,s))
    else:
        start_time = time.perf_counter()
        AX0 = A @ X0
        timing_A += time.perf_counter() - start_time
        tot_matvec_A += X0.shape[1]
        Rcurr = B - AX0
        X = X0
    
    if K is not None: X -= K @ (K.T @ X) # remove part from kernel

    # Metrics
    abs_conv_characteristics = []
    rel_conv_characteristics = []
    abs_two_conv_characteristics = []
    rel_two_conv_characteristics = []
    residual_norms = []
    block_sizes = []

    # Initial metrics
    block_sizes.append(Rcurr.shape[1])
    residual_norms.append( np.linalg.norm(Rcurr, ord="fro") )
    elem_rel_residual_norms = (np.linalg.norm(Rcurr, axis=0)/B_norms)[:,None]
    if Xtrue is not None:
        abs_term, rel_term, elem_A_err = calc_conv_characteristic(A, X, Xtrue, norm="A")
        elem_A_errs = elem_A_err[:,None]
        abs_conv_characteristics.append( abs_term )
        rel_conv_characteristics.append( rel_term )
        abs_term, rel_term, elem_two_err = calc_conv_characteristic(A, X, Xtrue, norm="two")
        elem_two_errs = elem_two_err[:,None]
        abs_two_conv_characteristics.append( abs_term )
        rel_two_conv_characteristics.append( rel_term )


    # Check if already converged?
    if all_residuals_converged(Rcurr, B, tol=tol, B_norms=B_norms):
        converged = True
        info = {
            "n_iters": n_iters,
            "residual_norms": np.asarray(residual_norms),
            "converged": converged,
            "block_sizes": np.asarray(block_sizes),
            "singular_breakdown": singular_breakdown,
            "elem_rel_residual_norms": elem_rel_residual_norms,
            "tot_time": tot_time,
            "timing_A": timing_A,
            "timing_M": timing_M,
            "tot_matvec_A": tot_matvec_A,
            "tot_matvec_M": tot_matvec_M,
        }
        

        if Xtrue is not None:
            info["abs_conv_characteristics"] = np.asarray(abs_conv_characteristics)
            info["rel_conv_characteristics"] = np.asarray(rel_conv_characteristics)
        
        return X, info
    
    
    # Compute initial Z
    if M is not None: 
        start_time = time.perf_counter()
        Zcurr = M @ Rcurr
        timing_M = time.perf_counter() - start_time
        tot_matvec_M += Rcurr.shape[1]
    else:
        Zcurr = Rcurr
    
    if K is not None: Zcurr -= K @ (K.T @ Zcurr) # remove part from kernel

    # Compute initial P
    if variant == "HS":
        P = Zcurr
    elif variant == "DP":
        P, _ = np.linalg.qr(Zcurr, mode='reduced')
    elif variant == "BF":
        P, _, _, _, _ = chan_rrqr(Zcurr, tau=tau)
    block_sizes.append(P.shape[1])

    # Begin iteration
    converged = False
    singular_breakdown = False

    for k in range(maxiter):

        if variant == "HS":
            start_time = time.perf_counter()
            Q = A @ P
            timing_A += time.perf_counter() - start_time
            tot_matvec_A += P.shape[1]
            PtQ = P.T @ Q
            ZcurrtRcurr = Zcurr.T @ Rcurr

            # Try to solve, break if singular
            try:
                Alpha = np.linalg.solve(PtQ, ZcurrtRcurr)
                cond = np.linalg.cond(PtQ)
                if cond > 1e16: # break if condition number is massive
                    singular_breakdown
                    break
            except:
                singular_breakdown = True
                break

            X += P @ Alpha
            Rnext = Rcurr - ( Q @ Alpha )
            if K is not None: 
                X -= K @ (K.T @ X) # remove part from kernel
                Rnext -= K @ (K.T @ Rnext)
            Rstopping = Rnext

        else:
            start_time = time.perf_counter()
            Q = A @ P
            timing_A += time.perf_counter() - start_time
            tot_matvec_A += P.shape[1]
            PtQ = P.T @ Q
            Gamma = np.linalg.solve(PtQ, P.T @ Rcurr)
            X += P @ Gamma
            Rcurr -= Q @ Gamma
            if K is not None: 
                X -= K @ (K.T @ X) # remove part from kernel
                Rcurr -= K @ (K.T @ Rcurr)
            Rstopping = Rcurr

        # Metrics?
        residual_norms.append( np.linalg.norm(Rstopping, ord="fro") )
        elem_rel_residual_norms = np.hstack([ elem_rel_residual_norms, (np.linalg.norm(Rstopping, axis=0)/B_norms)[:,None] ])
        if Xtrue is not None:
            abs_term, rel_term, elem_A_err = calc_conv_characteristic(A, X, Xtrue, norm="A")
            elem_A_errs = np.hstack([elem_A_errs, elem_A_err[:,None] ])
            abs_conv_characteristics.append( abs_term )
            rel_conv_characteristics.append( rel_term )
            abs_term, rel_term, elem_two_err = calc_conv_characteristic(A, X, Xtrue, norm="two")
            elem_two_errs = np.hstack([elem_two_errs, elem_two_err[:,None] ])
            abs_two_conv_characteristics.append( abs_term )
            rel_two_conv_characteristics.append( rel_term )

        # Check stopping criteria?
        n_iters += 1
        if all_residuals_converged(Rstopping, B, tol=tol, B_norms=B_norms):
            converged = True
            break
        else:
            pass


        # Proceed with iteration if necessary
        if variant == "HS":
            if M is not None:
                start_time = time.perf_counter()
                Znext = M @ Rnext
                timing_M = time.perf_counter() - start_time
                tot_matvec_M += Rnext.shape[1]
            else:
                Znext = Rnext

            if K is not None:  Znext -= K @ (K.T @ Znext) # remove part from kernel
    
            ZnexttRnext = Znext.T @ Rnext
            Beta = np.linalg.solve(ZcurrtRcurr, ZnexttRnext)
            P = Znext + ( P @ Beta )
            if K is not None:  P -= K @ (K.T @ P) # remove part from kernel

            # Advance curr to next
            Zcurr = Znext
            ZcurrtRcurr = ZnexttRnext
            Rcurr = Rnext

        else:
            if M is not None:
                start_time = time.perf_counter()
                Zcurr = M @ Rcurr
                timing_M = time.perf_counter() - start_time
                tot_matvec_M += Rcurr.shape[1]
            else:
                Zcurr = Rcurr
        
            Delta = - np.linalg.solve( PtQ, Q.T @ Zcurr  )
            J = Zcurr + P @ Delta
            if K is not None:  J -= K @ (K.T @ J) # remove part from kernel

            if variant == "BF":
                P, _, _, rank, _ = chan_rrqr(J, tau=tau)
                P = P[:,:rank]
            elif variant == "DP":
                P, _ = np.linalg.qr(J, mode='reduced')
            
        block_sizes.append(P.shape[1]) # track the block size

        # If deflated to nothing, break
        if P.shape[1] == 0:
            break
        
    tot_end_time = time.perf_counter()
    tot_time = tot_end_time - tot_start_time

    info = {
        "n_iters": n_iters,
        "residual_norms": np.asarray(residual_norms),
        "converged": converged,
        "block_sizes": np.asarray(block_sizes),
        "singular_breakdown": singular_breakdown,
        "elem_rel_residual_norms": elem_rel_residual_norms,
        "tot_time": tot_time,
        "timing_A": timing_A,
        "timing_M": timing_M,
        "tot_matvec_A": tot_matvec_A,
        "tot_matvec_M": tot_matvec_M,
    }

    if K is None:
        info["K_proj_norm"] = 0.0
    else:
        info["K_proj_norm"] = np.linalg.norm( K @ ( K.T @ X ), ord="fro" )

    if Xtrue is not None:
        info["abs_conv_characteristics"] = np.asarray(abs_conv_characteristics)
        info["rel_conv_characteristics"] = np.asarray(rel_conv_characteristics)
        info["abs_two_conv_characteristics"] = np.asarray(abs_two_conv_characteristics)
        info["rel_two_conv_characteristics"] = np.asarray(rel_two_conv_characteristics)
        info["elem_A_errs"] = elem_A_errs
        info["elem_two_errs"] = elem_two_errs

    return X, info













