import collections

import numpy

from sklearn import linear_model
from sklearn import model_selection
from sklearn import base

RegressionResult = collections.namedtuple('RegressionResult', ('y_est', 'resid', 'R2', 'regressor', 'X'))

def regress(X, y, C=None, regressor=None):
    """Perform a basic regression task with any of the scikit-learn regressors.

    Parameters:
        X: input data points. Shape must be either (n_data) or (n_data, n_features).
        y: output data points. Shape must be either (n_data) or (n_data, n_targets),
            for multi-target regression. (The latter is just a convenience over
            performing n_targets individual single-target regressions.)
        C: data points to control the X values for. Shape must be either (n_data) or
            (n_data, n_features). Controlling in this case means removing any
            relationship between the X values and the C values. Thus, we first
            use regression to determine how each feature in X relates to the C
            values. Then we subtract off this relationship (i.e. get the residuals),
            leaving the component of X that is unrelated to the trends in the C
            values.
        regressor: an instance of any sklearn regressor. If None, use
            sklearn.linear_model.LinearRegression()

    Returns:
        y_est: y-values estimated by regression against the input X.
        resid: residuals (i.e. y - y_est)
        R2: R^2 value, the fraction of the total variance in y that is explained
            by the variance in X.
        regressor: regressor used to calculate the fit. Useful for examining the
            fit parameters such as the coefficients.
        X: the input data. Useful if the X values were transformed by controlling
            for the C values.
    """
    X = numpy.asarray(X)
    y = numpy.asarray(y)
    if X.ndim == 1:
        X = X[:, numpy.newaxis]

    if regressor is None:
        regressor = linear_model.LinearRegression()

    if C is not None:
        # subtract off everything about the X values that can be predicted from the control data
        # note, most regressors can do multi-target prediction, so we can control for each of the
        # features in X simultaneously
        X -= _fit_predict(C, X, regressor)

    y_est = _fit_predict(X, y, regressor)
    resid = y - y_est
    R2 = 1 - (resid**2).mean(axis=0) / y.var(axis=0) # ratio is variance unexplained divided by total variance
    return RegressionResult(y_est, resid, R2, regressor, numpy.squeeze(X))


class CVRegress(base.BaseEstimator):
    """A sklearn-style regressor that uses cross-validation to prevent fitting
    on the same data used for prediction.

    This regressor will give an estimate of the overall performance of a
    given regression method on a new dataset, and will prevent overfitting.

    Parameters:
        regressor: any sklearn-style regressor, including nonlinear methods.
        cv: fold for K-fold cross-validation, or an sklearn-style cross-validation
            iterator, such as sklearn.model_selection.LeaveOneOut
    """
    def __init__(self, regressor, cv=10):
        self.regressor = regressor
        self.cv = cv

    def fit(self, X, y):
        # needed so that constructing a pipeline with CVRegress doesn't barf
        raise RuntimeError('Only call fit_predict for CVRegress')

    def fit_predict(self, X, y):
        return model_selection.cross_val_predict(self.regressor, X, y=y, cv=self.cv)


def _fit_predict(X, y, regressor):
    if hasattr(regressor, 'fit_predict'):
        return regressor.fit_predict(X, y)
    else:
        regressor.fit(X, y)
        return regressor.predict(X)