from musp import Location
import numpy as np
from math import pi

class ASTF:
    def produce_tf_for_location(self, location, rate):
        # return a FUNCTION of zero arguments that returns an (ASTF, block_size) tuple
        pass
    
class DelayASTF(ASTF):
    def __init__(self):
        pass

    def produce_tf_for_location(self, location, rate):
        def produce_astf():
            try:
                # Not useful to specify 3D in cartesian. If there are only two values, xy.
                # Also "backwards compatibility." Yeah.
                x, y = location
            except:
                # Living life in 3D.
                r, theta, phi = location
                # Whole plane is inclined by phi about ears axis, so only this x, y pair needed
                x = r*np.cos(theta)
                y = r*np.sin(theta)

            decays = np.array(zip(Location(location).decays_at_ears()))
            delays = np.array(zip(Location(location).delays_to_ears()))
            # use a relatively long block, half a second
            orig_filter_length = rate / 20
            # use 20% more samples than needed for the maximum delay
            overlap_samples = int(max(delays) * 1.2)
            overall_samples = orig_filter_length + overlap_samples
            # A shift of n is realized by a multiplication by exp(2pi*n*w/T)
            # (but it can be fractional!)
            exp_coeff = -2j * pi * rate / overall_samples
            transfer_func = decays * np.exp(exp_coeff * delays * \
                    np.tile(np.arange(overall_samples), (2, 1))) 
            return (transfer_func, orig_filter_length)
        return produce_astf



