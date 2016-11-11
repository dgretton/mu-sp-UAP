from numpy import arctan2, sin, cos, sqrt, pi
from math import log

class Location:
    ear_separation = .2
    # the distance at which a sound is imagined to be heard when it's at unit volume:
    standard_distance = .2
    c_sound = 344.0

    def __new__(typ, *args, **kwargs):
        # basically, when "casting" to a Location, if it already is one, leave it.
        # fine because locations are immutable
        if isinstance(args[0], Location):
            return args[0]
        return object.__new__(typ, *args, **kwargs)

    def __init__(self, *coordinates):
        def try_unwrap(coordinates):
            try:
                x, y, z = coordinates
                self._cartesian = (float(x), float(y), float(z)) 
                self._spherical = None
            except:
                (theta, phi), r = coordinates
                self._spherical = ((float(theta), float(phi)), float(r))
                self._cartesian = None
        try:
            try_unwrap(coordinates[0])
        except:
            try_unwrap(coordinates)
        self.eardists = None

    def spherical(self):
        if not self._spherical:
            x, y, z = self._cartesian
            s = sqrt(x*x + y*y)
            r = sqrt(s*s + z*z)
            self._spherical = (arctan2(x, y), arctan2(z, s)), r
        return self._spherical

    def cartesian(self):
        if not self._cartesian:
            ((theta, phi), r) = self._spherical
            # theta = phi = 0 is straight "ahead," at (0, 1, 0)
            self._cartesian = r*sin(theta)*cos(phi), r*cos(theta)*cos(phi), r*sin(phi)
        return self._cartesian

    def delays_to_ears(self):
        left_dist, right_dist = self.dists_to_ears()
        return left_dist/Location.c_sound, right_dist/Location.c_sound

    def decays_at_ears(self):
        left_dist, right_dist = self.dists_to_ears()
        return Location.standard_distance/left_dist, Location.standard_distance/right_dist

    def dists_to_ears(self):
        if not self.eardists:
            x, y, z = self.cartesian()
            left_dist = sqrt((x + Location.ear_separation / 2)**2 + y*y + z*z)
            right_dist = sqrt((x - Location.ear_separation / 2)**2 + y*y + z*z)
            self.eardists = left_dist, right_dist
        return self.eardists

    def cartesian_distance_to(self, other):
        return sqrt(sum([(cs - co)**2 for cs, co in zip(self, other)]))

    @staticmethod
    def cartesian_distance(l1, l2):
        return l2.cartesian_distance_to(l1)

    def norm(self):
        return self.cartesian_distance_to((0, 0, 0))

    def cosine_distance_to(self, other):
        return 1.0 - sum([cs*co for cs, co in zip(self, other)])/(self.norm()*other.norm())

    @staticmethod
    def cosine_distance(l1, l2):
        return l2.cosine_distance_to(l1)

    def cache_tag(self):
        (theta, phi), r = self.spherical()
        r_bin = int(log(r, 1.1))
        theta_bin = int(theta/pi*180/4) # 4-degree bins
        phi_bin = int(phi/pi*180/4)
        return '@' + str(r_bin) + '_' + str(theta_bin) + '_' + str(phi_bin)

    def right_half_plane(self):
        x, y, z = self
        return Location(abs(x), y, z)

    def __str__(self):
        return "AS loc " + str(tuple("%.3f" % c for c in self.cartesian()))

    def __iter__(self):
        for c in self.cartesian():
            yield c

    def __add__(self, other):
        return Location(*[cs + co for cs, co in zip(self.cartesian(), other.cartesian())])
    
    def __eq__(self, other):
        return self.cartesian_distance_to(other) < .0001

