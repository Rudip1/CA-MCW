# original Python def
#        :param center: (float, float)
#        :param radius: float
#        :param angles: (float, float), initial and final angle as viewed from center, in standard convention, radians
#        :param speeds: (float, float), initial and final angular velocity
#        :param orientation: 1 for ccw, -1 for cw

std_msgs/String		        type  # either 'arc' or 'straight'
geometry_msgs/Point 	        center  # currently we only use the x, y coords
float64         		radius
float64	        		initial_angle
float64 			terminal_angle
float64 			orientation  # should be 1 for ccw and -1 for cw
float64 			initial_speed  # angular velocity at the initial point/angle. should be non-negative
float64 			terminal_speed  # angular velocity at the terminal point/angle. should be non-negative
float64 			distance  # if type is straight, a positive float, ignored otherwise
