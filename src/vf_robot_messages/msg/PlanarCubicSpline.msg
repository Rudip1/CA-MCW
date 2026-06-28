std_msgs/Header header

# A planar spline needs a sequence of control times and for each interval a pair of degree-3 polynomials.
uint32                              num_points
float64[]                           control_times  # timestamp of control points, increasing order, first timestamp always 0
Cubic[]                             heading_coeffs    # one for each time interval
Cubic[]                             velocity_coeffs  # one for each time interval
