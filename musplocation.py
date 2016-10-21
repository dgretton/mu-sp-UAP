from numpy import arctan2, sin, cos, sqrt, pi
from math import log

class Location:
    ear_separation = .2
    # the distance at which a sound is imagined to be heard when it's at unit volume:
    standard_distance = .2
    c_sound = 344.0

    def __init__(self, *coordinates, **astf):
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
        if astf:
            if any([k != "astf" for k in astf]):
                print "That's not an argument you can use in a location, silly!"
                exit()
            self.astf = astf["astf"]
        else:
            self.astf = None
        self.eardists = None

    def spherical(self):
        if not self._spherical:
            x, y, z = self._cartesian
            s = sqrt(x*x + y*y)
            r = sqrt(s*s + z*z)
            self._spherical = (arctan2(y, x), arctan2(z, s)), r
        return self._spherical

    def cartesian(self):
        if not self._cartesian:
            ((theta, phi), r) = self._spherical
            self._cartesian = r*cos(theta)*cos(phi), r*sin(theta)*cos(phi), r*sin(phi)
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
            left_dist = sqrt((x + Location.ear_separation / 2)**2 + y*y)
            right_dist = sqrt((x - Location.ear_separation / 2)**2 + y*y)
            self.eardists = left_dist, right_dist
        return self.eardists

    def cache_tag(self):
        if self.astf:
            return self.astf.cache_tag(location)
        (theta, phi), r = self.spherical()
        r_bin = int(log(r, 1.1))
        theta_bin = int(theta/pi*180/4) # 4-degree bins
        phi_bin = int(phi/pi*180/4)
        return '@' + str(r_bin) + '_' + str(theta_bin) + '_' + str(phi_bin)

    def __str__(self):
        return "Aural space location at " + str(self.cartesian())

    def __iter__(self):
        for c in self.cartesian():
            yield c

    def __add__(self, other):
        return Location(*[cs + co for cs, co in zip(self.cartesian(), other.cartesian())])
        #xs, ys, zs = self.cartesian()
        #xo, yo, zo = other.cartesian()
        #return Location(xs + xo, ys + yo, zs + zo

