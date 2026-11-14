# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb

# MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series
# Classification

# https://arxiv.org/abs/2012.08791

from numba import njit, prange, vectorize
import numpy as np

# Input: flat PPV feature index, per-dilation feature counts
# Output: (dilation_index, kernel_index) or (-1, -1) if out of range
# Steps through each dilation's block of 84 * num_features_per_dilation[d] features to find which dilation and kernel produced this feature
@njit(cache = True)
def _locate_feature(feature_index, num_features_per_dilation):
    block_start = 0
    for dilation_index in range(len(num_features_per_dilation)):
        features_per_kernel = num_features_per_dilation[dilation_index]
        block_end = block_start + 84 * features_per_kernel
        if feature_index < block_end:
            kernel_index = (feature_index - block_start) // features_per_kernel
            return dilation_index, kernel_index
        block_start = block_end
    return -1, -1

# Input: series _X, dilation, kernel's +2 positions (index_0, index_1, index_2)
# Output: C (the convolution output)
# Computes one kernel's dilated convolution directly
@njit(fastmath = True, cache = True)
def _convolve(_X, dilation, index_0, index_1, index_2):
    input_length = len(_X)
    C = np.zeros(input_length, dtype = np.float32)
    for t in range(input_length):
        total = np.float32(0)
        for i in range(9):
            s = t + (i - 4) * dilation
            if 0 <= s < input_length:
                if i == index_0 or i == index_1 or i == index_2:
                    total += np.float32(2) * _X[s]
                else:
                    total -= _X[s]
        C[t] = total
    return C

# Input: dilation_index, kernel_index, dilation, input_length
# Output: (lo, hi), the time-step range [lo, hi) that PPV counts for this feature
# Even (dilation_index + kernel_index) uses the whole zero-padded output; Odd uses only the middle part where the kernel never touches the padding
@njit(cache = True)
def _valid_range(dilation_index, kernel_index, dilation, input_length):
    padding = ((9 - 1) * dilation) // 2
    if (dilation_index + kernel_index) % 2 == 0:
        return 0, input_length
    return padding, input_length - padding

# Input: input_length, dilations, num_features_per_dilation, quantiles, num_pairs and seed
# Output: pair_a, pair_b, lags, widths
# Randomly choose the feature pairs used for ordered co-activation pooling (POC) and their corresponding lags and widths (window size)
def _fit_pairs(input_length, dilations, num_features_per_dilation, quantiles, num_pairs, seed):
    NUM_KERNELS = 84
    KERNEL_LENGTH = 9
    MIN_QUANTILE = 0.3
    LAG_TO_WIDTH_RATIO = 8

    longest_lag = input_length // 2

    block_size_per_dilation = NUM_KERNELS * num_features_per_dilation
    dilation_of_feature = np.repeat(dilations, block_size_per_dilation)

    kernel_span_of_feature = (KERNEL_LENGTH - 1) * dilation_of_feature # number of time steps between a kernel's first and last weight

    # decides which features are allowed to be used as pairs
    is_sparse = quantiles > MIN_QUANTILE
    is_short = kernel_span_of_feature < longest_lag
    candidates = np.where(is_sparse & is_short)[0]

    if num_pairs == 0 or len(candidates) < 2:
        no_pairs = np.zeros(0, dtype = np.int32)
        return no_pairs, no_pairs, no_pairs, no_pairs

    rng = np.random.default_rng(seed)

    pair_a = rng.choice(candidates, num_pairs)
    pair_b = rng.choice(candidates, num_pairs)

    # features never paired with itself
    needs_redraw = pair_a == pair_b
    while needs_redraw.any():
        pair_b[needs_redraw] = rng.choice(candidates, needs_redraw.sum())
        needs_redraw = pair_a == pair_b

    # shortest distance the search window's centre is created after a
    shortest_lag = np.maximum(kernel_span_of_feature[pair_a], kernel_span_of_feature[pair_b])

    # random choice as to how far the 'lag' a is from b, log-uniform between shortest_lag and longest_lag
    log_lags = rng.uniform(np.log(shortest_lag), np.log(longest_lag))
    lags = np.round(np.exp(log_lags)).astype(np.int32)

    # timing tolerance either side of the lag, window for b to appear
    widths = np.maximum(lags // LAG_TO_WIDTH_RATIO, 1).astype(np.int32)

    return pair_a.astype(np.int32), pair_b.astype(np.int32), lags, widths

@njit("float32[:](float32[:,:],int32[:],int32[:],float32[:])", fastmath = True, parallel = False, cache = True)
def _fit_biases(X, dilations, num_features_per_dilation, quantiles):

    num_examples, input_length = X.shape

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0,1,2,0,1,3,0,1,4,0,1,5,0,1,6,0,1,7,0,1,8,
        0,2,3,0,2,4,0,2,5,0,2,6,0,2,7,0,2,8,0,3,4,
        0,3,5,0,3,6,0,3,7,0,3,8,0,4,5,0,4,6,0,4,7,
        0,4,8,0,5,6,0,5,7,0,5,8,0,6,7,0,6,8,0,7,8,
        1,2,3,1,2,4,1,2,5,1,2,6,1,2,7,1,2,8,1,3,4,
        1,3,5,1,3,6,1,3,7,1,3,8,1,4,5,1,4,6,1,4,7,
        1,4,8,1,5,6,1,5,7,1,5,8,1,6,7,1,6,8,1,7,8,
        2,3,4,2,3,5,2,3,6,2,3,7,2,3,8,2,4,5,2,4,6,
        2,4,7,2,4,8,2,5,6,2,5,7,2,5,8,2,6,7,2,6,8,
        2,7,8,3,4,5,3,4,6,3,4,7,3,4,8,3,5,6,3,5,7,
        3,5,8,3,6,7,3,6,8,3,7,8,4,5,6,4,5,7,4,5,8,
        4,6,7,4,6,8,4,7,8,5,6,7,5,6,8,5,7,8,6,7,8
    ), dtype = np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    biases = np.zeros(num_features, dtype = np.float32)

    feature_index_start = 0

    for dilation_index in range(num_dilations):

        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):

            feature_index_end = feature_index_start + num_features_this_dilation

            _X = X[np.random.randint(num_examples)]

            A = -_X          # A = alpha * X = -X
            G = _X + _X + _X # G = gamma * X = 3X

            C_alpha = np.zeros(input_length, dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

            biases[feature_index_start:feature_index_end] = np.quantile(C, quantiles[feature_index_start:feature_index_end])

            feature_index_start = feature_index_end

    return biases

def _fit_dilations(input_length, num_features, max_dilations_per_kernel):

    num_kernels = 84

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (9 - 1))
    dilations, num_features_per_dilation = \
    np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base = 2).astype(np.int32), return_counts = True)
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32) # this is a vector

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation

# low-discrepancy sequence to assign quantiles to kernel/dilation combinations
def _quantiles(n):
    return np.array([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype = np.float32)

def fit(X, num_features = 10_000, max_dilations_per_kernel = 32, num_pairs = 2_000, seed = None):

    _, input_length = X.shape

    num_kernels = 84

    dilations, num_features_per_dilation = _fit_dilations(input_length, num_features, max_dilations_per_kernel)

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    biases = _fit_biases(X, dilations, num_features_per_dilation, quantiles)

    # new pairs with their window widths and lags provided by _fit_pairs
    pair_a, pair_b, lags, widths = _fit_pairs(input_length, dilations, num_features_per_dilation, quantiles, num_pairs, seed)

    return dilations, num_features_per_dilation, biases, pair_a, pair_b, lags, widths

# _PPV(C, b).mean() returns PPV for vector C (convolution output) and scalar b (bias)
@vectorize("float32(float32,float32)", nopython = True, cache = True)
def _PPV(a, b):
    if a > b:
        return 1
    else:
        return 0

@njit("float32[:,:](float32[:,:],Tuple((int32[:],int32[:],float32[:],int32[:],int32[:],int32[:],int32[:])))", fastmath = True, parallel = True, cache = True)
def transform(X, parameters):

    num_examples, input_length = X.shape

    dilations, num_features_per_dilation, biases, pair_a, pair_b, lags, widths = parameters

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0,1,2,0,1,3,0,1,4,0,1,5,0,1,6,0,1,7,0,1,8,
        0,2,3,0,2,4,0,2,5,0,2,6,0,2,7,0,2,8,0,3,4,
        0,3,5,0,3,6,0,3,7,0,3,8,0,4,5,0,4,6,0,4,7,
        0,4,8,0,5,6,0,5,7,0,5,8,0,6,7,0,6,8,0,7,8,
        1,2,3,1,2,4,1,2,5,1,2,6,1,2,7,1,2,8,1,3,4,
        1,3,5,1,3,6,1,3,7,1,3,8,1,4,5,1,4,6,1,4,7,
        1,4,8,1,5,6,1,5,7,1,5,8,1,6,7,1,6,8,1,7,8,
        2,3,4,2,3,5,2,3,6,2,3,7,2,3,8,2,4,5,2,4,6,
        2,4,7,2,4,8,2,5,6,2,5,7,2,5,8,2,6,7,2,6,8,
        2,7,8,3,4,5,3,4,6,3,4,7,3,4,8,3,5,6,3,5,7,
        3,5,8,3,6,7,3,6,8,3,7,8,4,5,6,4,5,7,4,5,8,
        4,6,7,4,6,8,4,7,8,5,6,7,5,6,8,5,7,8,6,7,8
    ), dtype = np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    num_pairs = len(pair_a)

    features = np.zeros((num_examples, num_features + num_pairs), dtype = np.float32)

    for example_index in prange(num_examples):

        _X = X[example_index]

        A = -_X          # A = alpha * X = -X
        G = _X + _X + _X # G = gamma * X = 3X

        feature_index_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros(input_length, dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        features[example_index, feature_index_start + feature_count] = _PPV(C, biases[feature_index_start + feature_count]).mean()
                else:
                    for feature_count in range(num_features_this_dilation):
                        features[example_index, feature_index_start + feature_count] = _PPV(C[padding:-padding], biases[feature_index_start + feature_count]).mean()

                feature_index_start = feature_index_end
        
        # POC code block
        # for each pair, the fraction of valid time steps 't' where 'a' fires at 't' and 'b' fires within [t + lag - width, t + lag + width]
        # computed by recomputing a's and b's convolutions
        for pair_index in range(num_pairs):

            feature_a = pair_a[pair_index]
            feature_b = pair_b[pair_index]
            lag = lags[pair_index]
            width = widths[pair_index]

            # which kernel and dilation each feature belongs to
            dilation_index_a, kernel_index_a = _locate_feature(feature_a, num_features_per_dilation)
            dilation_index_b, kernel_index_b = _locate_feature(feature_b, num_features_per_dilation)
            dilation_a = dilations[dilation_index_a]
            dilation_b = dilations[dilation_index_b]

            # their convolution outputs on this series
            C_a = _convolve(_X, dilation_a, indices[kernel_index_a, 0], indices[kernel_index_a, 1], indices[kernel_index_a, 2])
            C_b = _convolve(_X, dilation_b, indices[kernel_index_b, 0], indices[kernel_index_b, 1], indices[kernel_index_b, 2])

            # the time steps PPV would count for each feature
            lo_a, hi_a = _valid_range(dilation_index_a, kernel_index_a, dilation_a, input_length)
            lo_b, hi_b = _valid_range(dilation_index_b, kernel_index_b, dilation_b, input_length)

            # t is valid if a is defined at t and b is defined at t + lag
            t_start = max(lo_a, lo_b - lag)
            t_end = min(hi_a, hi_b - lag)

            if t_end <= t_start:
                continue

            bias_a = biases[feature_a]
            bias_b = biases[feature_b]

            matches = 0
            for t in range(t_start, t_end):
                if C_a[t] > bias_a:  # a fires at t
                    u_start = max(lo_b, t + lag - width)
                    u_end = min(hi_b, t + lag + width + 1)
                    for u in range(u_start, u_end):
                        if C_b[u] > bias_b:  # b fires in the window
                            matches += 1
                            break

            features[example_index, num_features + pair_index] = matches / (t_end - t_start)
    return features
