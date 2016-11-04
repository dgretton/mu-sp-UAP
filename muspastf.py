import os, json, atexit
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, pi
from mpl_toolkits.mplot3d import Axes3D
from musp import Location

class ASTF:

    def __init__(self, astf_data_generator, location, post_processor=None, filename=None):
        self.location = location
        # Generator goes to get the data we want to save to disk (maybe from disk), and
        # post-processor does some lightweight manipulation of the loaded data before returning it.
        # (you wouldn't want to save angle-dependent transfer functions that have been scaled
        # for distance, right?)
        self.astf_data_generator = astf_data_generator
        self.post_processor = (lambda l,d,irl:(d,irl)) if post_processor is None else post_processor
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

    def apply_decays(self, astf_data, location, start_location=None):
        decays = np.array(zip(Location(location).decays_at_ears()))
        if start_location is not None:
            decays /= np.array(zip(start_location.decays_at_ears()))
        return astf_data * decays

    def correct_delays(self, astf_data, location, max_delay_samples=None, start_location=None):
        delays = np.array(zip(Location(location).delays_to_ears()))*self.rate
        if start_location is not None:
            delays -= np.array(zip(start_location.delays_to_ears()))*self.rate
        # delays = np.clip(delays, 0, float('inf')) # neg delay shift ok if correctly cached
        if max_delay_samples is not None:
            print "Before compressing delays:", delays
            def compression_func(x):
                return np.where(x>0, x/(exp(x) + x), x)
            delay_fracs = delays/max_delay_samples
            delays = max_delay_samples*compression_func(delay_fracs)
            print "After compressing delays:", delays
        data_len = astf_data.shape[1]
        overall_samples = (data_len - 1)*2
        # A shift of n is realized by a multiplication by exp(2pi*n*w/T) (but it can be fractional!)
        exp_coeff = -2j * pi / overall_samples
        transfer_func = exp(exp_coeff * delays * \
                np.tile(np.arange(data_len), (2, 1))) 
        return astf_data * transfer_func

class EarDelayAuralSpace(AuralSpace):

    def _create_astf(self, location):
        # use a relatively long block, within an order of a second
        block_samples = int(self.rate / 200)
        # use 20% more samples than needed for the maximum delay
        delayr, delayl = location.delays_to_ears()
        impulse_samples = int(max(10, delayr*self.rate, delayl*self.rate)*1.2)

        def edas_astf_generator():
            tabula_rasa = np.ones((2, (block_samples + impulse_samples)/2 + 1))
            delayed = self.correct_delays(tabula_rasa, location)
            decayed = self.apply_decays(delayed, location)
            return decayed, impulse_samples

        return ASTF(edas_astf_generator, location)


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
                    astf_data_generator = lambda f=filepath: (np.load(f), ir_length)
                    x, y, z = meta[DiscreteAuralSpace.json_loc_property]
                    location = Location(float(x), float(y), float(z))
                    print "AAAAH WE LOADED AN ASTF!!"
                    self.astfs.append(ASTF(astf_data_generator, location, filename))
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
        nearest_post = [ASTF(a.astf_data_generator, location,
            post_processor=self.wrapped_as._astf_post_processor(location), filename=a.filename) for
            a in nearest]
        return nearest_post

    def _create_astf(self, location):
        print "That discrete aural space got no way to make new astfs out of thin air..."
        exit()

    def _astf_post_processor(self, destination_location):
        # default astf post processor applies shifts and decays assuming that the loaded astf
        # is at the standard distance; override if not
        def vanilla_post_processor(loc, data_from_cache, ir_len):
            filter_length = (data_from_cache.shape[1] - 1)*2
            delayed = self.correct_delays(data_from_cache, destination_location,
                    max_delay_samples=(filter_length - ir_len), start_location=loc)
            decayed = self.apply_decays(delayed, destination_location, start_location=loc)
            return (decayed, ir_len)
        return vanilla_post_processor

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
                print "So that's it."
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
    num_points = 100
    points = [Location((pi*2/num_points*(i-num_points/2), 0), Location.standard_distance)
            for i in range(num_points)]

    def n_nearest_locations(self, n, location):
        return sorted(DiscreteEarDelayAS.points, key=lambda l:l.cosine_distance_to(location))[:n]



discretized_delay_as = lambda rate: DiscreteAuralSpace("ddas", rate,
        wrapped_as_class=DiscreteEarDelayAS)

KEMAR_aural_space = lambda rate: DiscreteAuralSpace("KEMAR", rate)
