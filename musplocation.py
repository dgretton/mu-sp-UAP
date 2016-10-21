from numpy import arctan2, sin, cos, sqrt

class Location:

    def __init__(self, *coordinates, **astf):
        try:
            x, y, z = coordinates
            self._cartesian = (float(x), float(y), float(z)) 
            self._spherical = None
        except:
            (theta, phi), r = coordinates
            self._spherical = ((float(theta), float(phi)), float(r))
            self._cartesian = None
        print astf
        self.astf = astf

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
            self._cartesian = r*cos(theta), r*sin(theta), r*sin(phi)
        return self._cartesian

    def __iter__(self):
        for c in self.spherical():
            yield c

    def __add__(self, other):
        return Location(*[cs + co for cs, co in zip(self.cartesian(), other.cartesian())])
        #xs, ys, zs = self.cartesian()
        #xo, yo, zo = other.cartesian()
        #return Location(xs + xo, ys + yo, zs + zo

