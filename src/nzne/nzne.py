from functools import partial
from typing import Any 

import numpy as np
import scipy 

def get_dip_size(fidelities: list[float], noise_strengths: list[float], fidelity_threshold: float=0.99):
    """
    Determine the dip size for the non-zero noise extrapolation.

    Parameters
    ----------
    fidelities : list[float]
        List of fidelities.
    noise_strengths : list[float]
        List of noise strengths.
    fidelity_threshold : float
        The fidelity threshold.

    Returns
    -------
    dip_size : float
        The dip size.
    """
    # Default dip size: the whole window
    dip_size = max(noise_strengths) - min(noise_strengths)

    for idx, f in enumerate(fidelities[1:]):
        if f >= fidelity_threshold:
            dip_size = noise_strengths[idx + 1] - min(noise_strengths)
            break

    return dip_size

def get_weights(fidelities: list[float], noise_strengths: list[float], target_noise_strength: float, dip_size: float, delta_f: float, delta_d: float):
    """
    Compute the weights for the non-zero noise extrapolation.

    Parameters
    ----------
    fidelities : list[float]
        List of fidelities.
    noise_strengths : list[float]
        List of noise strengths.
    target_noise_strength : float
        The target noise strength.
    dip_size : float
        The dip size.
    delta_f : float
        The fidelity exponent.
    delta_d : float
        The noise strength exponent.

    Returns
    -------
    weights : list[float]
        The weights.
    """
    weights = []
    if np.isclose(dip_size, 0.):
        dip_size = max(noise_strengths) - min(noise_strengths) 

    if np.isclose(dip_size, 0.):
        raise ValueError("Dip size close to zero. Abandon extrapolation.")
    
    for fidelity, noise_strength in zip(fidelities, noise_strengths):
        w = (fidelity**delta_f) * np.exp(-delta_d * np.abs(noise_strength - target_noise_strength) / dip_size)
        weights.append(w)
    return weights 


def bond_dim_extrapolate(fidelities: list[float], expvals: list[float], target_fidelity: float=1, num_points: int=2):
    """
    Performs bond dim extrapolation (actually does the extrapolation in fidelity).

    It returns a flag indicating whether or not the extrapolation was a good one. Let E_i be the expectation value
    of the observable at fidelity F_i (sorting fidelities as F_0 <= F_1 <= ... <= F_k). Then if

    | (E_k - E_{k-1}) / (F_k - F_{k-1}) | * (1 - F_k) >= |E_k| / 2,

    i.e. if the extrapolation moves further than |E_k| / 2 away from the final data point, we say that the extrapolation
    is a bad one.

    Returns
    -------
    extrapolated_value
        The extrapolated value at target_fidelity.
    extrapolation
        The extrapolation itself as a tuple of x and y coordinates.
    good_extrapolation
        A flag indicating whether the extrapolation was good or not.
    """
    def _f(x, a, b):
        return a + b * x

    extr_x, extr_y = fidelities[-num_points:], expvals[-num_points:]

    # Determine if the extrapolation is good or not
    if not np.isclose(expvals[-1], 0) and not np.isclose(fidelities[-1] - fidelities[-2], 0):
        good_extrapolation = abs((expvals[-1] - expvals[-2]) / (fidelities[-1] - fidelities[-2])) * (1 - fidelities[-1]) < abs(expvals[-1]) / 2
    else:
        good_extrapolation = True  # This will be filtered out by another criteria

    b, a = np.polyfit(extr_x, extr_y, deg=1)
    x_linspace = np.linspace(min(*fidelities, target_fidelity), max(*fidelities, target_fidelity))
    return _f(target_fidelity, a, b), (x_linspace, _f(x_linspace, a, b), ), good_extrapolation


def get_final_expvals(fidelities_and_expvals: dict[float, dict[str, list[float]]], ignore_sign_condition: bool = False, ignore_extrapolation_condition: bool = False, ignore_close_to_zero_condition: bool = False, num_points: int = 2) -> dict[float, dict[str, Any]]:
    """
    Given a dictionary of fidelities and expvals, indexed by noise strength, filter the data points according to three criteria.

    The dictionary should have the form
    {
        noise_strength: {
            fidelities: [...],
            expvals: [...],
        }
    }

    First, a bond dimension extrapolation is performed to obtain extrapolated expectation values. Then the criteria are applied.

    The three criteria are:
    - C1: Remove data points whose (extrapolated) value is smaller than 1e-10.
    - C2: Remove data points that came from a bad extrapolation (see the bond_dim_extrapolate() function).
    - C3: Remove data points with a different sign to the pure-state data point.

    Returns
    -------
    final_data
        A new dictionary of noise strengths and expectation values.
    meta_data
        A dictionary containing all data, updated to store things like extrapolations, satisficaiton of criteria, etc.
    """
    extrapolated_points = {}
    metadata = {}
    for noise_strength in sorted(list(fidelities_and_expvals.keys())):
        data = fidelities_and_expvals[noise_strength]
        
        fidelities = data["fidelities"]
        expvals = data["expvals"]
        bond_dims = data["bond_dims"]
        
        # Remove nan values (set them to 0 for stability -- they will be dealt with later)
        expvals = np.nan_to_num(expvals)

        extrapolated_point, extrapolation, good_extrapolation = bond_dim_extrapolate(fidelities, expvals, num_points=num_points)

        C1 = C2 = C3 = True

        # C2: Skip bad extrapolations
        if not good_extrapolation:
            C2 = False or ignore_extrapolation_condition
        
        # C1: Skip small expectation values
        # if np.abs(extrapolated_point) <= 1e-10:
        if np.abs(extrapolated_point) <= 1e-17:
            C1 = False or ignore_close_to_zero_condition

        # C3: Skip points with differing signs from the pure data point
        if noise_strength > 0 and np.sign(extrapolated_point) != np.sign(extrapolated_points[0.0]):
            C3 = False or ignore_sign_condition
        
        # If we pass all of C1, C2, and C3, keep the data point
        if C1 and C2 and C3 or noise_strength == 0.0:
            extrapolated_points[noise_strength] = extrapolated_point
        
        # Store metadata
        metadata[noise_strength] = {
            "fidelities": fidelities,
            "expvals": expvals,
            "bond_dims": bond_dims,
            "extrapolated_point": extrapolated_point,
            "extrapolation": extrapolation,
            "C1 (non-zero)": C1,
            "C2 (good extrapolation)": C2,
            "C3 (same sign as pure-state)": C3, 
        }

    return extrapolated_points, metadata


def weighted_exponential_fit(noise_strengths: list[float], expvals: list[float], weights: list[float]):
    """
    Return a weighted exponential fit of expvals and noise strengths.

    Precisely, return coefficients a and b such that 
    log(y) = a + bx 
    is a good fit of the expvals (y) against noise_strengths (x).

    Also returns the log_expvals for plotting purposes.
    """
    # Start by taking the log of all expectation values
    log_expvals = [np.log(e) for e in expvals]

    # Fit a straight line
    coeffs = np.polyfit(noise_strengths, log_expvals, deg=1, w=weights)

    # return b, a, log_expvals
    return coeffs[0], coeffs[1], log_expvals


def curve_fit_weighted_exponential_fit(noise_strengths: list[float], expvals: list[float], weights: list[float], fun, p0):
    """
    Return a weighted exponential fit of expvals and noise strengths after shifting.

    Precisely, return coefficients a and b such that 
    log(y) = a + bx 
    is a good fit of the expvals (y) against noise_strengths (x).

    Also returns the log_expvals for plotting purposes.
    """
    noise_strengths = np.asarray(noise_strengths)
    expvals = np.asarray(expvals)
    weights = np.asarray(weights)
    p0 = np.asarray(p0)

    def weighted_least_squares(coeffs, fun, x, y, weights):
        z = fun(x, *coeffs)
        return np.sum([w**2 * (_z - _y)**2 for w, _z, _y in zip(weights, z, y)])

    optimize_fun = partial(weighted_least_squares, fun=fun, x=noise_strengths, y=expvals, weights=weights)

    res = scipy.optimize.minimize(optimize_fun, x0=p0, method="Nelder-Mead")

    return res.x


def nonzero_noise_extrapolation(noise_strengths: list[float], expvals: list[float], weights: list[float], target_noise_strength: float):
    """
    Perform non-zero noise extrapolation via a log-linear fit.

    Parameters
    ----------
    noise_strengths : list[float]
        List of noise strengths.
    expvals : list[float]
        List of expectation values corresponding to the noise strengths.
    target_noise_strength : float
        The noise strength to extrapolate the expectation value to.

    Returns
    -------
    extrapolated_expval : float
        The extrapolated expectation value at the target noise strength.
    """
    def _f(x, a, b):
        return a + b * x

    # Make a weighted fit
    b, a, log_expvals = weighted_exponential_fit(noise_strengths, expvals, weights)

    # Get the extrapolated log(expval)
    extrapolated_log_expval = _f(target_noise_strength, a, b)

    # Make line for plotting
    x = np.linspace(min(noise_strengths), max(noise_strengths), 1000)
    y = _f(x, a, b)

    return extrapolated_log_expval, log_expvals, (x, y)


def curve_fit_nonzero_noise_extrapolation(noise_strengths: list[float], expvals: list[float], weights: list[float], target_noise_strength: float):
    """
    Perform non-zero noise extrapolation by fitting directly to an exponential ansatz.

    Parameters
    ----------
    noise_strengths : list[float]
        List of noise strengths.
    expvals : list[float]
        List of expectation values corresponding to the noise strengths.
    weights : list[float]
        List of weights for the curve fitting.
    target_noise_strength : float
        The noise strength to extrapolate the expectation value to.

    Returns
    -------
    extrapolated_log_expval : float
        The extrapolated (log) of the expectation value at the target noise strength.
    extrapolation_plot : tuple
        A tuple containing the x and y values of the fitted curve for plotting.
    """
    # _f = a e^{b * t} + c
    def _f(t, a, b, c):
        return a * np.exp(b * t) + c
    
    p0 = [-1., -1., 1.]

    # Make a weighted fit
    a, b, c = curve_fit_weighted_exponential_fit(noise_strengths, expvals, weights, fun=_f, p0=p0)

    # Get the extrapolated log(expval)
    extrapolated_log_expval = _f(target_noise_strength, a, b, c)

    # Make line for plotting
    x = np.linspace(min(noise_strengths), max(noise_strengths), 1000)
    y = _f(x, a, b, c)

    return extrapolated_log_expval, (x, y)


# Full non-zero noise extrapolation pipeline for a single expectation value
def full_non_zero_noise_extrapolation_pipeline(
        fidelities_and_expvals_by_noise_strength: dict[float, float], 
        target_noise_strength: float, 
        fidelity_threshold: float=0.99, 
        ignore_pure_state: bool=False, 
        outlier_noise_strengths: list[float]=[]
        ):
    """
    Perform a full non-zero noise extrapolation pipeline for a single expectation value.

    The pipeline consists of the following steps:
    1. Perform bond dimension extrapolation and filter data points.
    2. Compute the dip size.
    3. Compute the weights.
    4. Perform the non-zero noise extrapolation.

    Parameters
    ----------
    fidelities_and_expvals_by_noise_strength
        A dictionary of fidelities and expectation values and bond dimensions, indexed by noise strength.
        This should take the form of {<noise_strength>: data}, where data is a dictionary of the form
        {"fidelities": [...], "expvals": [...], "bond_dims": [...]}.
    target_noise_strength
        The target noise strength at which to extrapolate.
    fidelity_threshold
        The fidelity threshold for the dip size.
    ignore_pure_state
        Whether or not to ignore the pure state data point.
    outlier_noise_strengths
        A list of noise strengths to ignore.

    Returns
    -------
    extrapolated_value
        The extrapolated value at target_noise_strength.
    extrapolated_log_expval
        The extrapolated log(expval) at target_noise_strength.
    log_expvals
        The log(expval) values for plotting purposes.
    extrapolation_plot
        The extrapolation plot itself.
    sorted_noise_strengths
        The sorted noise strengths.
    """
    # Do bond dim extrapolations and filter data points
    filtered_data, metadata = get_final_expvals(fidelities_and_expvals_by_noise_strength, num_points=2) 

    if len(filtered_data) <= 2:
        raise ValueError("Need more than 2 points for extrapolation.")

    filtered_data = {ns: d for ns, d in filtered_data.items() if ns not in outlier_noise_strengths}

    clean_metadata = {}
    for _ns, _md in metadata.items():
        clean_metadata[_ns] = _md.copy()
        clean_metadata[_ns].pop("extrapolation")

    # Make sure everything is sorted by noise strength so that all the data is in a consistent order
    sorted_noise_strengths = sorted(list(filtered_data.keys()))
    sorted_expvals = [filtered_data[ns] for ns in sorted_noise_strengths]

    if any(np.isnan(s) or np.isinf(s) for s in sorted_expvals):
        raise ValueError("Oops")
    
    # Sort fidelities
    sorted_best_fidelities = [max(fidelities_and_expvals_by_noise_strength[ns]["fidelities"]) for ns in sorted_noise_strengths]

    # Compute dip size
    dip_size = get_dip_size(sorted_best_fidelities, sorted_noise_strengths, fidelity_threshold=fidelity_threshold)

    # Compute weights
    weights = get_weights(sorted_best_fidelities, sorted_noise_strengths, target_noise_strength, dip_size, delta_f=2, delta_d=20)
    
    if any(np.isnan(w) or np.isinf(w) for w in weights):
        raise ValueError("Oops")

    # Do non-zero noise extrapolation on the filtered data
    # First make all values positive if they are negative 
    assert all(np.sign(sorted_expvals[0]) == np.sign(s) for s in sorted_expvals)

    negated = False
    if sorted_expvals[0] < 0:
        sorted_expvals = [-1 * s for s in sorted_expvals]
        negated = True 
    
    # Ignore pure-state data point if asked for
    if ignore_pure_state and sorted_noise_strengths[0] == 0.0:
        print("Ignoring pure-state data point")
        sorted_noise_strengths = sorted_noise_strengths[1:]
        sorted_expvals = sorted_expvals[1:]
        weights = weights[1:]

    # Do extrapolation
    extrapolated_log_expval, log_expvals, extrapolation_plot = nonzero_noise_extrapolation(sorted_noise_strengths, sorted_expvals, weights, target_noise_strength)

    # Take exponential
    extrapolated_value = np.exp(extrapolated_log_expval)

    # Reverse sign if necessary
    if negated:
        extrapolated_value = -extrapolated_value

    return extrapolated_value, extrapolated_log_expval, log_expvals, extrapolation_plot, sorted_noise_strengths



def full_non_zero_noise_extrapolation_pipeline_exponential_fit(fidelities_and_expvals_by_noise_strength: dict[float, float], target_noise_strength: float, ignore_pure_state: bool=False, outlier_noise_strengths: list[float]=[]):
    """
    Perform a full non-zero noise extrapolation pipeline via fitting directly to an exponential ansatz (for a single expectation value).

    The pipeline consists of the following steps:
    1. Perform bond dimension extrapolation and filter data points.
    2. Perform the non-zero noise extrapolation.

    Parameters
    ----------
    fidelities_and_expvals_by_noise_strength
        A dictionary of fidelities and expectation values and bond dimensions, indexed by noise strength.
        This should take the form of {<noise_strength>: data}, where data is a dictionary of the form
        {"fidelities": [...], "expvals": [...], "bond_dims": [...]}.
    target_noise_strength
        The target noise strength at which to extrapolate.
    fidelity_threshold
        The fidelity threshold for the dip size.
    ignore_pure_state
        Whether or not to ignore the pure state data point.
    outlier_noise_strengths
        A list of noise strengths to ignore.

    Returns
    -------
    extrapolated_value
        The extrapolated value at target_noise_strength.
    extrapolated_log_expval
        The extrapolated log(expval) at target_noise_strength.
    log_expvals
        The log(expval) values for plotting purposes.
    extrapolation_plot
        The extrapolation plot itself.
    sorted_noise_strengths
        The sorted noise strengths.
    """
    # Do bond dim extrapolations and filter data points
    filtered_data, metadata = get_final_expvals(fidelities_and_expvals_by_noise_strength, num_points=2, ignore_sign_condition=True) 

    # Ignore any manually-specified data points
    filtered_data = {ns: d for ns, d in filtered_data.items() if ns not in outlier_noise_strengths}

    clean_metadata = {}
    for _ns, _md in metadata.items():
        clean_metadata[_ns] = _md.copy()
        clean_metadata[_ns].pop("extrapolation")

    # Make sure everything is sorted by noise strength so that all the data is in a consistent order
    sorted_noise_strengths = sorted(list(filtered_data.keys()))
    sorted_expvals = [filtered_data[ns] for ns in sorted_noise_strengths]

    # Default: unweighted
    weights = np.ones(len(sorted_expvals))

    # Ignore pure-state data point if asked for
    if ignore_pure_state and sorted_noise_strengths[0] == 0.0:
        sorted_noise_strengths = sorted_noise_strengths[1:]
        sorted_expvals = sorted_expvals[1:]
        weights = weights[1:]

    extrapolated_value, extrapolation_plot = curve_fit_nonzero_noise_extrapolation(sorted_noise_strengths, sorted_expvals, weights, target_noise_strength)

    return extrapolated_value, extrapolation_plot, sorted_noise_strengths, sorted_expvals
