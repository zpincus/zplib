import numpy

def survival_curve_from_lifespans(lifespans, uncertainty_interval=0):
    """Produce a survival curve from a set of lifespans with optional uncertainty.

    If no uncertainty interval is provided, produce a standard, stair-step survival
    curve (which indicates infinite precision in when the time of death is known).
    Alternately, if an uncertainty interval is provided, it is assumed that the
    provided lifespan is actually +/- half of the interval. (e.g. If lifespans
    were assayed every 5 hours, provide 5 as the uncertainty interval and give
    the single lifespan as the midpoint of the interval.) In this case a non-stair-
    step plot is produced to reflect a uniform distribution of possible death times
    in that interval. This latter mode is generally preferable.

    Parameters:
        lifespans: list of point estimates of the lifespans of individuals in a
            population. This should be the middle of the interval between
            lifespan measurements.
        uncertainty_interval: Length of time between lifespan measurements.

    Returns: t, f
        t: timepoints
        f: fraction alive at each timepoint (1 to 0)
    """
    if uncertainty_interval > 0:
        lifespans = numpy.asarray(lifespans)
        half_interval = uncertainty_interval / 2
        return survival_curve_from_intervals(lifespans - half_interval, lifespans + half_interval)
    else:
        timepoints = [0]
        fraction_alive = [1]
        step = 1/len(lifespans)
        for lifespan in lifespans:
            last_frac = fraction_alive[-1]
            timepoints.extend([lifespan, lifespan])
            fraction_alive.extend([last_frac, last_frac-step])
        return timepoints, fraction_alive

def survival_curve_from_intervals(last_alives, first_deads):
    """Produce a survival curve from a set of (last_alive, first_dead) timepoints.

    This function produces a survival curve for a population of individuals, using
    two timepoints per individual: the last known time the individual was alive,
    and the first known time it was dead. Curves plotted in this fashion do not
    have the stairstep shape of curves with point estimates of lifespans; moreover
    these curves more accurately represent the uncertainty of death times.

    Parameters:
        last_alives, first_deads: lists containing the last-known-alive timepoint
            and first-know-dead timepoint for each individual in the population.

    Returns: t, f
        t: timepoints
        f: fraction alive at each timepoint (1 to 0)
    """
    last_alives = list(last_alives)
    first_deads = list(first_deads)
    assert len(first_deads) == len(last_alives)
    timepoints = sorted([0] + last_alives + first_deads)
    number_alive = numpy.zeros(len(timepoints), float)
    # for each individual, construct a curve that is 1 before last_alive,
    # 0 after first_dead, and linear in between, and get the value of that
    # curve at every timepoint that will be in the final curve.
    # We then sum up all of these curves to get the total survival curve
    for last_alive, first_dead in zip(last_alives, first_deads):
        number_alive += numpy.interp(timepoints, [last_alive, first_dead], [1, 0])
    fraction_alive = number_alive / len(last_alives)
    return timepoints, fraction_alive