import numpy
import scipy.stats.kde as kde
import scipy.ndimage as ndimage

def viterbi(p_observations_given_state, p_transition, p_initial):
    """Given an HMM and set of observations, identify the most likely state
    sequence to have generated the observations using the Viterbi algorithm.

    This implements a generalization of the Viterbi algorithm where the
    transition probabilities may be specified in such a way as to change from
    observation to observation. (Formally, this could be modeled as a HMM with
    a new set of states for each observation, but that is more complex.) This
    allows for modeling of processes like lifespan where age can modify the
    probability of state transition (e.g. death).

    Paramters:
        (Let N be the number of observations and S the number of states.)
        p_observations_given_state: Array of shape (N, S), where element [t, s]
            is the probability of the observation at time t assuming the system
            was in fact in state s.
        p_transition: Array of shape (S, S), where element [si, sj] is the
            probability of transitioning from state si to state sj.
            Alternately, array of shape (N-1, S, S) where [t, si, sj] is the
            probability of transition from state si to state sj between time t
            and time t+1.
            Note that in either case, p_transition must sum to 1 across the sj's.
        p_initial: List of length S giving the proability of starting out in each
            state. Must sum to 1.

    Returns: list of length N giving the index of each state at each time, for
        the most likely path through the states.
    """
    p_observations_given_state = numpy.asarray(p_observations_given_state)
    p_transition = numpy.asarray(p_transition)
    p_initial = numpy.asarray(p_initial)
    N, S = p_observations_given_state.shape
    assert p_transition.shape in {(S, S), (N-1, S, S)}
    if p_transition.shape == (S, S):
        p_transition = numpy.array([p_transition for i in range(N-1)])
    assert numpy.allclose(numpy.sum(p_transition, axis=2), 1)
    assert p_initial.shape == (S,)
    assert numpy.allclose(numpy.sum(p_initial), 1)

    # convert all probabilities to log probabilities so we can sum instead of
    # multiplying, which better controls numerical error.
    err = numpy.seterr(divide='ignore') # allow log(0) to go to -inf, as desired
    lp_observations_given_state = numpy.log(p_observations_given_state)
    lp_transition = numpy.log(p_transition)
    lp_initial = numpy.log(p_initial)
    numpy.seterr(**err)

    states = numpy.arange(S)
    # path[i] always contains the maximum likelihood sequence of states ending at state i
    path = [[i] for i in states]
    # lp_state contains the current log probability of being in the state given the sequence
    # of observations thus far considered.
    lp_state = lp_observations_given_state[0] + lp_initial

    for lp_obs, lp_trans in zip(lp_observations_given_state[1:], lp_transition):
        # For each observation after the first timepoint, construct an (S, S)
        # shape array where [si, sj] contains the log probability of going from
        # state si to state sj between time t and t+1.
        # Assume we know for each state si prob(si at time t), the probability
        # of being in that state at that time, then we can calculate the probability
        # of being in any given state sj at time t+1:
        # prob(transition from si at time t to sj at time t+1) = prob(si at t) *
        #    prob(si->sj between t and t+1) *
        #    prob(observation at t+1 given state sj)
        # prob(j at time t+1) = max_i(prob(i at time t -> j at time t+1))
        #
        # Thus we merely need to keep updating our estimates for the probability
        # of being in each state at each time, and keep a list of the path that
        # lead to each state.
        #
        # The actual code in use  is 100% equivalent to the code below; however it
        # is rather more efficient.
        #
        # lp_transition_t = numpy.zeros((s, s), dtype=float)
        # new_path = []
        # lp_state = []
        # for s_to in states:
        #     best_from_lp = -numpy.inf
        #     for s_from in states:
        #         lp_transition_t[s_from, s_to] = lp_state[s_from] + lp_trans[s_from, s_to] + lp_obs[s_to]
        #         if lp_transition_t[s_from, s_to] > best_from_lp:
        #             best_from = s_from
        #             best_from_lp = lp_transition_t[s_from, s_to]
        #     lp_state.append(best_from_lp)
        #     new_path.append(path[best_from] + [s_to])
        # path = new_path
        lp_transition_t = lp_state[:,numpy.newaxis] + lp_trans + lp_obs[numpy.newaxis,:]
        best_from = numpy.argmax(lp_transition_t, axis=0)
        path = [path[s_from]+[s_to] for s_to, s_from in enumerate(best_from)]
        lp_state = lp_transition_t[best_from, states]
    last_state = numpy.argmax(lp_state)
    return numpy.array(path[last_state])

class ObservationProbabilityEstimator:
    def __init__(self, state_observations, continuous=True, pseudocount=1):
        """Given a set of observations known to belong to different states,
        construct an object that will estimate the probability that new observations
        belong to each state (i.e. the p_observations_given_state matrix)

        Parameters:
            state_observations: list of length S containing an array of observation
                data for each state.
            continuous: if True, the observation data is assumed to be continuous.
                If false, the data will be treated as categorical observations from
                0 to to the maximum value seen across any observation.
            pseudocount: if data are treated as categorical, add the specified
                pseudocount to each category, to prevent zero-probability estimates.
        """
        self.continuous = continuous
        state_observations = [numpy.asarray(so) for so in state_observations]
        if continuous:
            self.state_distributions = [kde.gaussian_kde(so) for so in state_observations]
        else:
            max_val = max(so.max() for so in state_observations)
            state_counts = [numpy.bincount(so, minlength=max_val) + pseudocount]
            self.state_distributions = [sc / sc.sum() for sc in state_counts]

    def __call__(self, observations):
        """Estimate the p_observations_given_state matrix for a set of observations.

        If observations is a list/array of length N, returns an array of shape
        (N, S), where element [t, s] is the probability of the observation at
        time t assuming the system was in fact in state s.
        """
        observations = numpy.asarray(observations)
        if self.continuous:
            state_probabilities = [kde(observations) for kde in self.state_distributions]
        else:
            state_probabilities = [hist[observations] for hist in self.state_distributions]
        return numpy.transpose(state_probabilities)

def estimate_hmm_params(state_sequences, pseudocount=1, moving=True, time_sigma=1):
    """Given a set of state sequences, estimate the initial and transition
    probabilities for each state (i.e. the p_initial and p_transition matrices
    needed for HMM inference).

    Parameters:
        state_sequences: array of shape (n,t) with timecourses of state
            assignments for n different sequences of measurements, each of
            length t.
        pseudocount: add specified pseudocount to each count (of initial state
            or transition probabilities), to keep any probability from being
            estimated as zero.
        moving: if True, the returned p_transition will be of shape (n-1, s, s),
            where n is as above and s is the number of states. This matrix will
            contain the probability of transitions at each observation time.
            If False, p_transition is of shape (s, s), and contains the grand
            transition probabilities over all times.
        time_sigma: if moving is True, smooth the probability distributions by
            a gaussian moving average with their neighbors with the specified
            sigma. To disable, set time_sigma to 0.

    Returns: p_initial, p_transition
    """
    state_sequences = numpy.asarray(state_sequences)
    n, t = state_sequences.shape
    s = state_sequences.max() + 1 # number of states
    initial_counts = numpy.bincount(state_sequences[:,0], minlength=s) + pseudocount
    p_initial = initial_counts / (n + s*pseudocount)
    p_transition = []
    for i in range(t-1):
        from_states = state_sequences[:, i]
        to_states = state_sequences[:, i+1]
        p_trans = []
        for from_s in range(s):
            from_mask = (from_states == from_s)
            tos = to_states[from_mask]
            p_trans.append(numpy.bincount(tos, minlength=s))
        p_transition.append(p_trans)
    p_transition = numpy.array(p_transition) # shape (n-1, s, s)
    if not moving:
        p_transition = p_transition.sum(axis=0) # shape (s, s)
    p_transition += pseudocount
    denom = p_transition.sum(axis=-1) # shape (n-1, s) or (s,)
    denom[denom == 0] = 1 # avoid 0/0 cases. Just set them to probability = 0 by converting to 0/1
    p_transition = p_transition / denom[...,numpy.newaxis]
    if moving and time_sigma:
        p_transition = ndimage.gaussian_filter1d(p_transition, time_sigma, axis=0, mode='nearest')
    return p_initial, p_transition