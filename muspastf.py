import os, json, atexit, copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
from numpy import exp, pi
from mpl_toolkits.mplot3d import Axes3D
from math import radians
from musp import Location

class ASTF:

    def __init__(self, astf_data_generator, location, post_processor=None, filename=None):
        self.location = location
        # Generator goes to get the data we want to save to disk (maybe from disk), and
        # post-processor does some lightweight manipulation of the loaded data before returning it.
        # (you wouldn't want to save angle-dependent transfer functions that have been scaled/
        # delayed for distance, right?)
        self.astf_data_generator = astf_data_generator
        self.post_processor = (lambda l,d,irl:(d,irl)) if post_processor is None else post_processor
        self.filename = filename
        self.data_c = ASTF.DataContainer()

    def generate_astf(self):
        if self.data_c.data is None:
            self.data_c.data, self.data_c.ir_length = self.astf_data_generator()
        return self.post_processor(self.location, self.data_c.data, self.data_c.ir_length)

    def with_post_processor(self, pp):
        new_astf = copy.copy(self) # retain data container, generator
        new_astf.post_processor = pp
        return new_astf

    def has_data(self):
        return self.data_c.data is not None

    def data(self):
        return self.data_c.data

    def ir_length(self):
        return self.data_c.ir_length

    def __str__(self):
        return "ASTF @" + str(self.location) + " with generator " + str(self.astf_data_generator) + \
             (" from " + self.filename if self.filename is not None else '') + \
             (" impulse " + str(self.data_c.ir_length) + " samples long with data @" + \
                 str(id(self.data_c.data)) if self.has_data() else '')

    class DataContainer:
        def __init__(self):
            self.data = None
            self.ir_length = None

        def __iter__(self):
            yield data
            yield self.ir_length
        

class AuralSpace(object):

    def astf_for_location(self, location):
        return self._create_astf(Location(location))

    def apply_decays(self, astf_data, location, start_location=None):
        decays = np.array(zip(Location(location).decays_at_ears()))
        print "Here be the decays:", decays
        if start_location is not None:
            decays /= np.array(zip(start_location.decays_at_ears()))
        return astf_data * decays

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([0, 0, 0], [0, 10, 0], linewidth=5)
    
    def correct_delays(self, astf_data, ir_length, location,
            max_delay_samples=None, start_location=None, ax=ax):
        x, y, z = location
        ax.scatter([x], [y], [z])
        delays = np.array(zip(Location(location).delays_to_ears()))*self.rate
        if start_location is not None:
            if start_location == location:
                return astf_data, ir_length
            # neg delay shift ok if correctly cached
            delays -= np.array(zip(Location(start_location).delays_to_ears()))*self.rate
            xs, ys, zs = start_location
            ax.scatter([xs], [ys], [zs])
            ax.plot([x, xs], [y, ys], [z, zs], linewidth=min(5, 5.0/(float(max(delays))*.01)))#color=(.1, .5, float(max(delays))*.001))

        if max_delay_samples is not None:
            def compression_func(x):
                return np.where(x>0, x/(exp(x) + x), x)
            delay_fracs = delays/max_delay_samples
            delays = max_delay_samples*compression_func(delay_fracs)
        data_len = astf_data.shape[1]
        overall_samples = (data_len - 1)*2
        # A shift of n is realized by a multiplication by exp(2pi*n*w/T) (but it can be fractional!)
        exp_coeff = -2j * pi / overall_samples
        transfer_func = exp(exp_coeff * delays * \
                np.tile(np.arange(data_len), (2, 1))) 
        return astf_data * transfer_func, int(ir_length + max(*delays))

class EarDelayAuralSpace(AuralSpace):

    def __init__(self, name, rate):
        self.name = name
        self.rate = rate

    def _create_astf(self, location):
        # use a relatively long block, within an order of a second
        block_samples = int(self.rate * .5)
        # use 20% more samples than needed for the maximum delay
        delayr, delayl = location.delays_to_ears()
        impulse_samples = int(max(delayr*self.rate, delayl*self.rate)*1.2)

        def edas_astf_generator():
            tabula_rasa = np.ones((2, (block_samples + impulse_samples)/2 + 1))
            delayed, mod_ir_length = self.correct_delays(tabula_rasa, impulse_samples,
                    location, max_delay_samples=block_samples)
            decayed = self.apply_decays(delayed, location)
            return decayed, mod_ir_length

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
                    astf_data_generator = lambda f=filepath, irl=ir_length: (np.load(f), irl)
                    x, y, z = meta[DiscreteAuralSpace.json_loc_property]
                    location = Location(float(x), float(y), float(z))
                    self.astfs.append(ASTF(astf_data_generator, location, filename))
        print "ALL OF THESE GREAT THINGS WERE ADDED RIGHT AT THE BEGINNING:"
        #for astf in self.astfs:
        #    print astf
        atexit.register(self._save_out_cache)

    def astf_for_location(self, location):
        astf, = self.wrapped_as.n_nearest_astfs(1, Location(location))
        return astf

    def n_nearest_locations(self, n, location):
        # default behavior only makes sense when whole cache is populated in advance
        nearest = sorted(self.astfs,
                key=lambda a: self.dist_metric(a.location, Location(location)))[:n]
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
        return [a.with_post_processor(self.wrapped_as._astf_post_processor(location))
                for a in nearest]

    def _astf_post_processor(self, destination_location):
        # default astf post processor applies shifts and decays assuming that the loaded astf
        # is at the standard distance; override if not
        def vanilla_post_processor(loc, data_from_cache, ir_len):
            filter_length = (data_from_cache.shape[1] - 1)*2
            delayed, mod_ir_len = self.correct_delays(data_from_cache, ir_len, destination_location,
                    max_delay_samples=(filter_length - ir_len), start_location=loc)
            decayed = self.apply_decays(delayed, destination_location, start_location=loc)
            return (decayed, mod_ir_len)
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
        plt.show()
        exit()
        print "SAVING OUT THE CACHE! YAY! IT WORKED!"
        if not self.astfs:
            print "...but there's nothing to save. aw." 
            return
        plot_things = []
        #for astf in self.astfs:
        #    print astf
        #    plot_things.append([c for c in astf.location])
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(*zip(*plot_things))
        #plt.show()
        if not os.path.exists(self.unique_cache_dir):
            print "making new cache directory for aural space " + self.name
            os.mkdir(self.unique_cache_dir)
        meta_map = {}
        for astf in self.astfs:
            if not astf.has_data():
                continue
            filename = astf.filename if astf.filename else self.wrapped_as._cache_name_for_astf(astf)
            filepath = os.path.join(self.unique_cache_dir, filename)
            if not os.path.exists(filepath):
                with open(filepath, 'w+') as tf_file:
                    np.save(tf_file, astf.data())
            meta_map[filename] = {DiscreteAuralSpace.json_loc_property:
                                        tuple(c for c in astf.location),
                                  DiscreteAuralSpace.json_ir_length_property:
                                        astf.ir_length()}
        with open(self.cache_metadata_file, 'w') as mdf:
            json.dump(meta_map, mdf)


class DiscreteEarDelayAS(DiscreteAuralSpace, EarDelayAuralSpace):
    from math import pi
    num_points = 100
    points = [Location((pi*2/num_points*(i-num_points/2), 0), Location.standard_distance)
            for i in range(num_points)]

    def n_nearest_locations(self, n, location):
        return sorted(DiscreteEarDelayAS.points,
                key=lambda l:l.cosine_distance_to(Location(location)))[:n]

#    def _astf_post_processor(self, location):
#        def null_post_processor(loc, data_from_cache, ir_len):
#            ret = np.zeros(np.fft.irfft(data_from_cache).shape)
#            ret[:,0] = 100
#            ret[:,1] = 0
#            ret /= np.sqrt(np.sum(ret**2)/2)
#            ret = np.fft.rfft(ret)
#            return (ret, ir_len)
#        return null_post_processor


class KemarAuralSpace(DiscreteAuralSpace):
    hrtf_dir = os.path.join(os.path.expanduser('~'), ".mu-sp", "hrtf", "kemar")
    hrtf_avg_energy = .6

    def __init__(self, name, rate):
        # init and load HRTFs from cache
        super(self.__class__, self).__init__(name, rate)
        self.files_for_locs = {}
        plot_things = []
        for filename in os.listdir(KemarAuralSpace.hrtf_dir):
            iH = filename.index('H')
            ie = filename.index('e')
            ia = filename.index('a')
            elevation_deg, attitude_deg = float(filename[iH + 1:ie]), -float(filename[ie + 1:ia])
            print elevation_deg, attitude_deg
            loc = Location((radians(attitude_deg), radians(elevation_deg)), .2)
            plot_things.append([c for c in loc])
            self.files_for_locs[loc.cache_tag()] = loc, filename
            mirror_loc = loc.right_half_plane()
            plot_things.append([c for c in mirror_loc])
            self.files_for_locs[mirror_loc.cache_tag()] = mirror_loc, filename

        # build all remaining HRTFs
        for loc, filename in self.files_for_locs.values():
            if not self._saved_astf_for_location(loc):
                self.astfs.append(self._create_astf(loc))

        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(*zip(*plot_things))

        #plt.show()

    def _create_astf(self, location):
        block_length = self.rate * .05
        loc, filename = self.files_for_locs[location.cache_tag()]
        if not loc == location:
            print "Something went terribly wrong."
            exit()
        path = os.path.join(KemarAuralSpace.hrtf_dir, filename)
        def kas_generate_astf():
            filerate, data = wavfile.read(path)
            raw_data = np.transpose(np.array(data))
            unit_size_data = raw_data.astype(np.float) / (2**15)
            energy = KemarAuralSpace.hrtf_avg_energy #np.sum(unit_size_data**2)/2.0 # avg energy between the two tracks
            print "THE ENERGY:", energy
            impulse_data = unit_size_data #/np.sqrt(energy) # normalize by RMS
            if not location.right_half_plane() == location:
                impulse_data = impulse_data[::-1,:] # flip left and right channels
            hrtf_data = np.fft.rfft(np.hstack((impulse_data, np.zeros((2, block_length)))))
            #plt.plot(impulse_data[0])
            #plt.show()
            return hrtf_data, impulse_data.shape[1]

        return ASTF(kas_generate_astf, location)


#    def _astf_post_processor(self, location):
#        def null_post_processor(loc, data_from_cache, ir_len):
#            return (data_from_cache*.1, ir_len)
#        return null_post_processor





KEMAR_aural_space = lambda rate: DiscreteAuralSpace("KEMAR", rate)
