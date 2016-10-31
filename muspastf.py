import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, json, atexit, random
from musp import Location
import numpy as np
from math import pi

class ASTF:

    def __init__(self, astf_data_generator, location, post_processor=lambda l, d, irl: (d, irl),
            filename=None):
        self.location = location
        # Generator goes to get the data we want to save to disk (maybe from disk), and
        # post-processor does some lightweight manipulation of the loaded data before getting it.
        # (you wouldn't want to save angle-dependent transfer functions that have been scaled
        # for distance, right?)
        self.astf_data_generator = astf_data_generator
        self.post_processor = post_processor
        self.filename = filename
        self.data = None
        self.ir_length = None

    def generate_astf(self):
        if self.data is None:
            self.data, self.ir_length = self.astf_data_generator()
        return self.post_processor(self.location, self.data, self.ir_length)

    def __str__(self):
        return "ASTF @" + str(self.location) + " with generator " + str(self.astf_data_generator) + \
             (" from " + self.filename if self.filename is not None else '') + \
             (" impulse " + str(self.ir_length) + " samples long" if self.ir_length else '') + \
             (" with data @" + str(id(self.data)) if self.data is not None else '')

class AuralSpace:

    def _create_astf(self, location):
        pass

class EarDelayAuralSpace(AuralSpace):

    def _create_astf(self, location):
        rate = self.rate
        decays = np.array(zip(Location(location).decays_at_ears()))
        delays = np.array(zip(Location(location).delays_to_ears()))
        # use a relatively long block, .1s
        impulse_response_length = rate * .1
        # use 20% more samples than needed for the maximum delay
        overlap_samples = int(max(delays) * 1.2)
        overall_samples = impulse_response_length + overlap_samples
        # A shift of n is realized by a multiplication by exp(2pi*n*w/T)
        # (but it can be fractional!)
        exp_coeff = -2j * pi * rate / overall_samples

        def produce_astf():
            transfer_func = decays * np.exp(exp_coeff * delays * \
                    np.tile(np.arange(overall_samples), (2, 1))) 
            return (transfer_func, impulse_response_length)

        return ASTF(produce_astf,  location)


class DiscreteAuralSpace(AuralSpace):

    cache_path = os.path.join(os.path.expanduser('~'), ".mu-sp", "cache")
    metadata_file_name = "meta.json"
    json_loc_property = "loc"
    json_ir_length_property = "ir_len"

    def __init__(self, name, rate, wrapped_as_class=None, dist_metric=Location.cosine_distance,
            existing_cache_dirs=[]):
        self.name = name
        self.astfs = []
        self.wrapped_as = self if wrapped_as_class is None else wrapped_as_class("Discrete AS " +
                name + "-wrapped " + wrapped_as_class.__name__, rate)
        self.changed_metadata = False
        self.dist_metric = dist_metric
        self.rate = rate
        self.unique_cache_dir = os.path.join(DiscreteAuralSpace.cache_path, name)
        if self.unique_cache_dir in existing_cache_dirs:
            print "Ya just can't make two discrete aural spaces with the same cache directory, y'see"
            exit()
        existing_cache_dirs.append(self.unique_cache_dir)
        self.cache_metadata_file = os.path.join(self.unique_cache_dir,
                DiscreteAuralSpace.metadata_file_name)
        if os.path.exists(self.cache_metadata_file):
            with open(self.cache_metadata_file, 'r') as mdf:
                for filename, meta in json.load(mdf).iteritems():
                    ir_length = int(meta[DiscreteAuralSpace.json_ir_length_property])
                    filepath = os.path.join(self.unique_cache_dir, filename)
                    generate_astf = lambda f=filepath: (np.load(f), ir_length)
                    x, y, z = meta[DiscreteAuralSpace.json_loc_property]
                    location = Location(float(x), float(y), float(z))
                    post_process = self.wrapped_as._post_process_astf_data
                    print "AAAAH WE LOADED AN ASTF!!"
                    self.astfs.append(ASTF(generate_astf, location, post_process, filename))
        print "ALL OF THESE GREAT THINGS WERE ADDED RIGHT AT THE BEGINNING:"
        for astf in self.astfs:
            print astf
        atexit.register(self._save_out_cache)

    def astf_for_location(self, location):
        astf, = self.wrapped_as.n_nearest_astfs(1, location)
        return astf

    def n_nearest_locations(self, n, location):
        # default behavior only makes sense when whole cache is populated in advance
        nearest = sorted(self.astfs, key=lambda a: self.dist_metric(a.location, location))[:n]
        return [astf.location for astf in nearest]

    def n_nearest_astfs(self, n, location):
        locs = self.wrapped_as.n_nearest_locations(n, location)
        nearest = []
        for loc in locs[:]:
            astf = self._saved_astf_for_location(loc)
            if astf is None:
                new_astf = self.wrapped_as._create_astf(loc)
                self.astfs.append(new_astf)
                nearest.append(new_astf)
            else:
                nearest.append(astf)
        return nearest

    def _create_astf(self, location):
        print "That discrete aural space got no way to make new astfs out of thin air..."
        exit()

    def _post_process_astf_data(self, location, data, ir_length):
        return data, ir_length

    def _saved_astf_for_location(self, location):
        for astf in self.astfs:
            if astf.location == location:
                return astf
        return None

    def _cache_name_for_astf(self, astf):
        (t, p), r = astf.location.spherical()
        return self.name + '_' + '_'.join(['%.3f'%c for c in [t, p, r]]) + ".astfdata"

    def _save_out_cache(self):
        print "SAVING OUT THE CACHE! YAY! IT WORKED!"
        if not self.astfs:
            print "...but there's nothing to save. :("
            return
        plot_things = []
        for astf in self.astfs:
            print astf
            plot_things.append([c for c in astf.location])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*zip(*plot_things))
        plt.show()
        if not os.path.exists(self.unique_cache_dir):
            print "making new cache directory for aural space " + self.name
            os.mkdir(self.unique_cache_dir)
        meta_map = {}
        for astf in self.astfs:
            if astf.data is None:
                continue
            filename = astf.filename if astf.filename else self.wrapped_as._cache_name_for_astf(astf)
            filepath = os.path.join(self.unique_cache_dir, filename)
            if not os.path.exists(filepath):
                with open(filepath, 'w+') as tf_file:
                    np.save(tf_file, astf.data)
            meta_map[filename] = {DiscreteAuralSpace.json_loc_property:
                                        tuple(c for c in astf.location),
                                  DiscreteAuralSpace.json_ir_length_property:
                                        astf.ir_length}
        with open(self.cache_metadata_file, 'w') as mdf:
            json.dump(meta_map, mdf)

class DiscreteEarDelayAS(EarDelayAuralSpace, DiscreteAuralSpace):
    from math import pi
    num_points = 10
    points = [Location((pi*2/num_points*(i-num_points/2), 0), 2) for i in range(num_points)]

    def n_nearest_locations(self, n, location):
        return sorted(DiscreteEarDelayAS.points, key=lambda l:l.cosine_distance_to(location))[:n]

    def _post_process_astf_data(self, location, data, ir_length):
        decays = np.array(zip(Location(location).decays_at_ears()))
        return decays * data, ir_length

discretized_delay_as = lambda rate: DiscreteAuralSpace("ddas", rate,
        wrapped_as_class=DiscreteEarDelayAS)

KEMAR_aural_space = lambda rate: DiscreteAuralSpace("KEMAR", rate)
