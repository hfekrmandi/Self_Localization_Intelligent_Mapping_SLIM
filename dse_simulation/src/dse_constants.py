#!/usr/bin/env python
"""
Constants
"""

EULER_ORDER = 'zyx'                 # The euler rotation order
EULER_ORDER_3D_OBS = 'z'                 # The euler rotation order
INF_MATRIX_INITIAL = 10000          # Initial information matrix (covariance) value. Equivalent to 0.01 standard deviation
INF_VECTOR_INITIAL = 0.01           # Initial information vector (states) value. x = Y^-1*y, equivalent to 1
MOTION_BASE_COVARIANCE = 0.000001   # Motion model covariance for velocity. Equivalent to 1 mm/sec standard deviation
