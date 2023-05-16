"""Code obtained in repository: https://github.com/gabalz/cvxreg"""

import numpy as np

from lib.cvxapp.utils.partition import max_affine_partition


# from src.utils.regression import max_affine_predict


class Estimator:
    """Abstract class of an estimator."""

    def __init__(self, train, predict, subgradient=None):
        self.train = train
        self.predict = predict
        self.subgradient = subgradient


class EstimatorModel:
    """Abstract class of the model of an estimator."""

    def __init__(self, weights):
        self.weights = weights


def squared_distance(left, right, axis=1):
    """Calculates the squared Euclidean norm of each row of a matrix.

    :param left: left data matrix (or vector)
    :param right: right data matrix (or vector)
    :param axis: distance calculation along this axis (defaults to distances along rows)
    :returns: vector of distances for each row of the matrix

    """
    diff = np.square(left - right)
    if len(diff.shape) == 1:
        return np.sum(diff)
    return np.sum(diff, axis=axis)


def shuffle_data(X, y):
    """Shuffle the samples.

    :param X: data matrix (each row is a sample)
    :param y: target vector
    :return: shuffled X, shuffled y, permutation
    """
    assert X.shape[0] == len(y)
    perm = np.random.permutation(len(y))
    return X[perm, :], y[perm], perm


def normalize_data(X, y, tol=1e-6):
    """Centering and scaling the data, and dropping dependent columns.

    :param X: data matrix (each row is a sample)
    :param y: target vector
    :param tol: tolerance for dropping dependent columns
    :return: transformed X, transformed y,
             eigenvector matrix,
             mean vector of X, scaling vector of X,
             mean of y, scaler of y

    """
    n, d = X.shape
    assert len(y) == n

    # Centralizing:
    x_mean = np.mean(X, 0)
    y_mean = np.mean(y)
    X = X - x_mean
    y = y - y_mean

    # SVD: X == np.dot(u[:, :V.shape[0]] * s, V)
    U, s, V = np.linalg.svd(X, full_matrices=False)

    # Downscaling:
    x_scale = max(1, s[0])
    s /= x_scale
    y_scale = max(1, np.max(np.abs(y)))
    y /= y_scale

    # Dropping dependent columns:
    depcols = np.where(s < tol)
    if len(depcols) > 0:
        U = np.delete(U, depcols, axis=1)
        s = np.delete(s, depcols)
        V = np.delete(V, depcols, axis=0)
    X = U * s

    return X, y, V, x_mean, x_scale, y_mean, y_scale


def cv_indices(npoints, ncvfolds):
    """Calculating cross-validation indices.

    :param npoints: number of data points
    :param ncvfolds: number of CV-folds
    :return sample indices of the test and train sets, respectively

    """
    assert npoints >= ncvfolds
    idxset = set(range(npoints))
    test = []
    train = []
    val = 0
    step = npoints / ncvfolds
    for f in range(ncvfolds):
        next_val = val + step if f < ncvfolds - 1 else npoints
        idx = np.array(range(int(val), int(next_val)), dtype=int)
        test.append(idx)
        train.append(np.array(sorted(idxset - set(idx)), dtype=int))
        val = next_val
    return test, train


def _prepare_prediction(weights, X, extend_X1):
    if isinstance(weights, EstimatorModel):
        weights = weights.weights
    if extend_X1:
        X = np.insert(X, 0, 1.0, axis=1)
    return weights, X


def max_affine_predict(weights, X, extend_X1=True):
    """Prediction by a max-affine model.

    :param weights: max-affine weights (each row is a hyperplane), or EstimatorModel having the weights
    :param X: data matrix (each row is a sample)
    :param extend_X1: whether or not to extend the data with leading 1s
    :return: predicted vector (one value for each sample)
    """
    weights, X = _prepare_prediction(weights, X, extend_X1)
    return np.max(X.dot(weights.T), axis=1)


def max_affine_subgradient(weights, X, extend_X1=True):
    weights, X = _prepare_prediction(weights, X, extend_X1)
    activated_weight = weights[np.argmax(X.dot(weights.T), axis=1), :]
    return activated_weight


def _amap_loss(y, yhat):
    """Squared loss between two vectors.

    :param y: target vector
    :param yhat: estimate vector
    :return: squared loss
    """
    return squared_distance(y, yhat, axis=0)


def _amap_fit(X, y, betaI):
    """Ridge regression fitting.

    :param X: data matrix (each row is a sample augmented with a leading 1)
    :param y: target vector
    :betaI: identity matrix scaled by the regularization constant
    :return: weight vector of the ridge regressor
    """
    assert len(y) > 0
    assert X.shape[0] == len(y)
    return np.linalg.solve(X.T.dot(X) + betaI, X.T.dot(y))


def _calc_amap(X, y, P, W, XW, ins_err, mincellsize, betaI):
    """Internal AMAP calculation within a CV-fold."""
    K = len(P)
    d = X.shape[1] - 1

    # Find the best candidate split.

    best_err = ins_err
    best_k = best_i1 = best_i2 = best_w1 = best_w2 = best_Xw1 = best_Xw2 = None
    for k in range(K):
        cell = P[k]
        ncell = len(cell)
        if ncell < 2 * mincellsize:
            continue

        Xcell = X[cell, :]
        ycell = y[cell]
        XWk = XW[:, k]
        XW[:, k] = -np.inf
        maxXWk = XW
        if len(maxXWk.shape) > 1:
            maxXWk = np.max(maxXWk, axis=1)
        XW[:, k] = XWk

        for dim in range(1, d + 1):
            Xcelldim = Xcell[:, dim]
            split = np.median(Xcelldim)
            i1 = np.where(Xcelldim < split)[0]
            i2 = np.where(Xcelldim > split)[0]
            ni1 = len(i1)
            ni2 = len(i2)
            if ni1 + ni2 < ncell:  # merging the ties
                i3 = np.where(Xcelldim == split)[0]
                if ni1 < mincellsize:
                    i1 = np.concatenate((i1, i3[:ni1 - mincellsize]))
                    i3 = i3[ni1 - mincellsize:]
                elif ni2 < mincellsize:
                    i2 = np.concatenate((i2, i3[:ni2 - mincellsize]))
                    i3 = i3[ni2 - mincellsize:]
                ni3 = len(i3)
                if ni3 > 0:
                    split = int(np.round(ni3 / 2))
                    i1 = np.concatenate((i1, i3[:split]))
                    i2 = np.concatenate((i2, i3[split:]))
            assert len(i1) >= mincellsize
            assert len(i2) >= mincellsize

            w1 = _amap_fit(Xcell[i1, :], ycell[i1], betaI)
            w2 = _amap_fit(Xcell[i2, :], ycell[i2], betaI)
            Xw1 = X.dot(w1)
            Xw2 = X.dot(w2)
            cand_y = np.maximum(np.maximum(Xw1, Xw2), maxXWk)
            cand_err = _amap_loss(y, cand_y)
            if cand_err < best_err:
                best_k = k
                best_i1 = i1
                best_i2 = i2
                best_w1 = w1
                best_w2 = w2
                best_Xw1 = Xw1
                best_Xw2 = Xw2
                best_err = cand_err

    if best_k is None:
        return P, W, XW, ins_err

    # Fit the new partition.

    cell = P[best_k]
    P[best_k] = cell[best_i1]
    P[k] = cell[best_i2]
    W[:, best_k] = best_w1
    W = np.insert(W, K, best_w2, axis=1)
    XW[:, best_k] = best_Xw1
    XW = np.insert(XW, K, best_Xw2, axis=1)
    err = best_err
    K = K + 1
    assert W.shape[1] == K

    while True:
        # Compute the induced partition.
        cand_P = max_affine_partition(X, W.T)

        # Stop when minimum cell size requirement is violated.
        if min(cand_P.cell_sizes()) < mincellsize:
            break

        # Fit the induced partition.
        cand_P = cand_P.cells
        cand_W = np.zeros((d + 1, len(cand_P)))
        for k in range(len(cand_P)):
            cell = cand_P[k]
            cand_W[:, k] = _amap_fit(X[cell, :], y[cell], betaI)
        cand_XW = X.dot(cand_W)
        cand_err = _amap_loss(y, np.max(cand_XW, axis=1))

        # Update partition or stop.
        if cand_err < err:
            P = cand_P
            W = cand_W
            XW = cand_XW
            err = cand_err
        else:
            break

    return list(P), W, XW, err


class AMAPEstimatorModel(EstimatorModel):
    """The model of an AMAP estimator."""

    def __init__(self, weights, cv_errs, niter):
        EstimatorModel.__init__(self, weights)
        self.cv_errs = cv_errs
        self.niter = niter


def amap_train(
        X,
        y,
        ncvfolds=10,
        shuffle=True,
        mincellsize=None,
        nfinalsteps=5,
        ridge_regularizer=1e-6,
        cv_tol=1e-6,
        norm_data=True,
        dep_tol=1e-6,
        **kwargs
):
    """Training the AMAP estimator.

    :param X: data matrix (each row is a sample)
    :param y: target vector
    :param ncvfolds: number of cross-validation folds
    :param shuffle: whether or not to shuffle the data
    :param mincellsize: minimum number of elements within a cell
    :param nfinalsteps: number of steps after non-improving CV-error beyond tolerance
    :param ridge_regularizer: ridge regression regularization parameter
    :param cv_tol: tolerance for CV-error improvement
    :param norm_data: whether or not to normalize the data
    :param dep_tol: tolerance for dropping dependent data columns
    :return: weights, CV-errors, and number of iteration
    """
    n, d = X.shape
    assert n >= d, 'Too few data points!'
    assert ncvfolds >= 2
    if mincellsize is None:
        mincellsize = max(d + 1, np.ceil(np.log2(n)))

    # Preparing the data.

    assert len(y.shape) == 1 or y.shape[1] == 1
    if len(y.shape) > 1:
        y = y[:, 0]

    if shuffle:
        X, y, _ = shuffle_data(X, y)

    if norm_data:
        X, y, V, x_mean, x_scale, y_mean, y_scale = normalize_data(X, y, dep_tol)

    n, d = X.shape
    X = np.insert(X, 0, 1.0, axis=1)  # augmenting the data with leading 1s
    betaI = ridge_regularizer * np.eye(d + 1)
    betaI[0, 0] = 0.0  # do not regularize the bias term
    tlimit = np.ceil(n ** (d / (d + 4)))

    # Initialization.

    cv_test_idx, cv_train_idx = cv_indices(n, ncvfolds)
    P = []  # partition
    W = []  # weights, dim: (1+d) x K
    XW = []  # X*W
    ins_err = np.zeros(ncvfolds)  # in-sample errors
    outs_err = np.zeros(ncvfolds)  # out-of-sample errors
    for fold in range(ncvfolds):
        idx = cv_train_idx[fold]
        assert len(idx) > 0
        P.append([np.array(range(len(idx)))])
        W.append(np.array([_amap_fit(X[idx, :], y[idx], betaI)]).T)
        XW.append(X[idx, :].dot(W[fold]))
        ins_err[fold] = _amap_loss(y[idx], XW[fold][:, 0])

        idx = cv_test_idx[fold]
        outs_err[fold] = _amap_loss(y[idx], X[idx, :].dot(W[fold])[:, 0])

    best_cv_err = np.mean(outs_err)
    best_W = W

    # Cross-validation training.

    niter = 0
    maxiter = min(nfinalsteps, tlimit)
    outs_err *= 0.0
    while niter < maxiter:
        niter += 1
        for fold in range(ncvfolds):
            if len(P[fold]) < niter:
                continue

            idx = cv_train_idx[fold]
            P[fold], W[fold], XW[fold], ins_err[fold] = _calc_amap(
                X[idx, :], y[idx],
                P[fold], W[fold], XW[fold], ins_err[fold],
                mincellsize, betaI,
            )
            idx = cv_test_idx[fold]
            outs_err[fold] = _amap_loss(y[idx], np.max(X[idx, :].dot(W[fold]), axis=1))

        cv_err = np.mean(outs_err)
        if cv_err < best_cv_err - cv_tol:
            best_cv_err = cv_err
            best_W = W
            maxiter = min(niter + nfinalsteps, tlimit)

    # Choosing the final model.

    W = best_W
    best_err = np.inf
    best_W = None
    cv_errs = []
    for fold in range(ncvfolds):
        err = _amap_loss(y, np.max(X.dot(W[fold]), axis=1))
        cv_errs.append(err)
        if err < best_err:
            best_err = err
            best_W = W[fold]

    K = best_W.shape[1]
    b = best_W[0, :].T  # dim: K x 1
    W = best_W[1:, :].T  # dim: K x d
    if norm_data:
        W = W.dot(V) * (y_scale / x_scale)
        b = y_mean + b * y_scale - W.dot(x_mean.T)

    weights = np.zeros((K, 1 + d))
    weights[:, 0] = b
    weights[:, 1:] = W

    return AMAPEstimatorModel(weights, cv_errs, niter)


class AMAPEstimator:
    """The AMAP estimator"""

    def __init__(self):
        self.model = None

    def fit(self, X, y, **kwargs):
        self.model = amap_train(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return max_affine_predict(self.model, X, **kwargs)

    def subgradient(self, X, **kwargs):
        return max_affine_subgradient(self.model, X, **kwargs)
