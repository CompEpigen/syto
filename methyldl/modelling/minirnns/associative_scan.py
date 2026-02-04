# https://github.com/glassroom/heinsen_sequence
def associative_scan_log(log_coeffs, log_values, return_log=False):
    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
    log_h = a_star + log_h0_plus_b_star

    if return_log:
        return log_h
    else:
        return log_h.exp()
