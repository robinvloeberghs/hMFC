"""Constants of the model.
"""

A_MAX = 0.995 # upper limit for a_i
SIGMA_A_MAX = 0.2 # upper limit for standard deviation of truncated normal for a_i
SIGMA_A_MIN = 0.01 # lower limit for standard deviation of truncated normal for a_i (to steer it away from problematic zero region)

PG_TRUNC = 200 # number of gamma samples for PG sample
SIGMASQ0 = 1.0 # error variance x on first trial