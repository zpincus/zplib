import numpy
import scipy.stats as stats

def robust_mean_std(data, subset_fraction=0.7):
    """robust_mean_std(data, subset_fraction) -> (mean, std)

    Compute the mean and standard deviation of the fraction of the input data
    (as determined by subset_fraction) that has the minimal varianc. Thus, this
    procedure is robust to an outlier rate of up to (1-subset_fraction).

    Internally the univariate MCD estimator is used. To determine the exact data
    points that are judged to be outliers vs. inliers, use unimcd() directly.
    """
    data = numpy.asarray(data).flatten()
    n = len(data)
    h = int(round(n * subset_fraction))
    subset_mask = unimcd(data, h)
    inliers = data[subset_mask]
    return inliers.mean(), inliers.std()


def unimcd(y, h):
  """unimcd(y, h) ->  subset_mask

  unimcd computes the MCD estimator of a univariate data set. This estimator is
  given by the subset of h observations with smallest variance. The MCD location
  estimate is then the mean of those h points, and the MCD scale estimate is
  their standard deviation.

  A boolean mask is returned indicating which elements of the input array are
  in the MCD subset.

   The MCD method was introduced in:

   Rousseeuw, P.J. (1984), "Least Median of Squares Regression,"
   Journal of the American Statistical Association, Vol. 79, pp. 871-881.

   The algorithm to compute the univariate MCD is described in

   Rousseeuw, P.J., Leroy, A., (1988), "Robust Regression and Outlier
   Detection," John Wiley, New York.

  This function based on UNIMCD from LIBRA: the Matlab Library for Robust
  Analysis, available at: http://wis.kuleuven.be/stat/robust.html
  """
  y = numpy.asarray(y, dtype=float).flatten()
  ncas = len(y)
  length = ncas-h+1
  if length <= 1:
    return numpy.ones(len(y), dtype=bool)
  indices = y.argsort()
  y = y[indices]
  ind = numpy.arange(length-1)
  ay = numpy.empty(length)
  ay[0] = y[0:h].sum()
  ay[1:] = y[ind+h] - y[ind]
  ay = numpy.add.accumulate(ay)
  ay2=ay**2/h
  sq = numpy.empty(length)
  sq[0] = (y[0:h]**2).sum() - ay2[0]
  sq[1:] = y[ind+h]**2 - y[ind]**2 + ay2[ind] - ay2[ind+1]
  sq = numpy.add.accumulate(sq)
  sqmin=sq.min()
  ii = numpy.where(sq==sqmin)[0]
  Hopt = indices[ii[0]:ii[0]+h]
  ndup = len(ii)
  slutn = ay[ii]
  initmean=slutn[int(numpy.floor((ndup+1)/2 - 1))]/h
  initcov=sqmin/(h-1)
  # calculating consistency factor
  res=(y-initmean)**2/initcov
  sortres=numpy.sort(res)
  factor=sortres[h-1]/stats.chi2.ppf(float(h)/ncas,1)
  initcov=factor*initcov
  res=(y-initmean)**2/initcov #raw_robdist^2
  quantile=stats.chi2.ppf(0.975,1)
  weights=res<quantile
  weights=weights[indices.argsort()] #rew_weights
  return weights
