import wave, struct, os, math, contextlib, warnings, pyaudio
import numpy as np
import scipy.io.wavfile as wavfile
from itertools import chain
import matplotlib.pyplot as plt
from math import log, atan2, sqrt, pi

class Beat:

    def __init__(self, size=1.0, parent=None, label=None, seqnum=None):
        if parent is self:
            print "Wait... a beat was initialized with itself as a parent..."
            exit()
        self.label = label if label \
                else parent.label + "_" + str(seqnum) if parent \
                else "top"
        self.size = size
        self.beats = []
        self.unconstrained_beats = []
        self.parent = parent
        self._duration = None
        self._time = None
        self.linked_beats = []
        self.sounds = []

    def split(self, portions):
        try:
            # interpret an integer argument as a number of unconstrained beats
            portions = [None]*portions
        except:
            pass
        if len(portions) < 2:
            print "NO NO NO can't split a beat into zero or one pieces"
            return
        if self.beats != []:
            print "NOPE NOPE NOPE can't split nonempty beat"
            return
        total = sum([0.0 if p is None else float(p) for p in portions])
        self.beats = [Beat((portion/total if portion else None),  self, seqnum=i) \
                for i, portion in enumerate(portions)]
        self.unconstrained_beats = [b for b in self.beats if not b.size]
        if self.unconstrained_beats:
            print self.label, "has the following unconstrained beats:", \
                    ', '.join(map(str, self.unconstrained_beats))
        return self.beats

    def split_even(self, num):
        return self.split([1.0]*num)

    def interleave_split(self, times_map, total_time=1.0):
        sequence = []
        for label, times in times_map.iteritems():
            sequence += [(time, label) for time in times]
        sequence.sort(key=lambda x: x[0])
        portions = [t2[0] - t1[0] for t1, t2 in 
                zip(sequence, sequence[1:] + [(total_time, None)])]
        beats_for_labels = {}
        for beat, (t, label) in zip(self.split(portions), sequence):
            beats_for_labels[label] = beats_for_labels.get(label, [])
            beats_for_labels[label].append(beat)
        return beats_for_labels

    def link(self, lunk):
        if lunk is self:
            print "You probably didn't mean to link a beat to itself..."
            exit()
        if lunk not in self.linked_beats:
            print "linked", self, "to", lunk
            if lunk.has_duration():
                self.set_duration(lunk.duration())
            self.linked_beats.append(lunk)
            lunk.link(self)
            for linked in self.linked_beats:
                # I just wanted this to happen
                if linked is not lunk:
                    linked.link(lunk)
                    lunk.link(linked)

    def unconstrained_remaining(self):
        return [uc_b for uc_b in self.unconstrained_beats
                if not uc_b.has_duration()]

    def constrained(self):
        # This should be the test for whether a child should assign this beat
        # a duration outright.
        return len(self.unconstrained_remaining()) == 0

    def overconstrained_tree(self):
        print "Tree overconstraint error. Loosen up a little!"
        gdi = 1.0/0

    def underconstrained_tree(self):
        print "There wasn't enough to go on to define times and " \
                + "durations. Let's tighten this up."
        gdi = 2.0/0

    def set_duration(self, dur, recurse=True):
        if self.has_duration():
            if self._duration - dur >= .0001:
                print "Double-assigned a beat's duration somewhere."
                self.overconstrained_tree()
            else:
                return
        self._duration = dur
        if recurse:
            self.enact_local_constraints()
        for link in self.linked_beats:
            print "linked across to another tree..."
            link._duration = dur
            link.enact_local_constraints()
        return dur

    def has_duration(self):
        return self._duration is not None

    def enact_local_constraints(self):
        parent = self.parent
        if self.has_duration() and self.size and parent \
                and parent.constrained():
            print self, "setting duration of parent", parent
            parent.set_duration(self._duration/self.size + \
                    parent.unconstrained_duration())
            return

        dof_list = self.unconstrained_remaining()
        if not self.has_duration():
            dof_list.append(self)
        c_beats = [b for b in self.beats if b not in self.unconstrained_beats]

        if c_beats and not all([c_beat.has_duration() for c_beat in c_beats]):
            defined_beat = None
            for c_beat in c_beats:
                if c_beat.has_duration():
                    defined_beat = c_beat
                    break
            # if there's a (hopefully just one) beat with a duration, force the rest
            if defined_beat:
                c_beat_shared_dur = defined_beat.duration(True)/defined_beat.size
                for forced_beat in c_beats:
                    if forced_beat is defined_beat:
                        continue
                    print defined_beat, "forcing sibling beat", forced_beat
                    # quietly assign durations to siblings, no higher recursion
                    forced_beat.set_duration(c_beat_shared_dur*forced_beat.size,
                            recurse=False)
            # otherwise, all of them are unassigned even though they're present, so
            # they'll need to be assigned using time remaining after unconstrained
            else:
                dof_list.append(c_beats)

        if len(dof_list) == 0:
            print self, "fully constrained"
        elif len(dof_list) == 1:
            last_dof, = dof_list
            if last_dof is self:
                self.set_duration(sum([b.duration(True) for b in self.beats]))
            elif last_dof is c_beats:
                c_beat_shared_dur = self.duration(True) - sum([b.duration(True) for \
                        b in self.unconstrained_beats])
                for forced_beat in c_beats:
                    forced_beat._duration = c_beat_shared_dur*forced_beat.size
            else:
                other_unconstrained = self.unconstrained_beats[:].remove(last_dof)
                last_dof.set_duration(self.duration(True) - sum([ou.duration(True)
                    for ou in other_unconstrained]))
            self.enact_local_constraints()
        else:
            print "waiting on", ','.join([str(dof) for dof in dof_list])

        if parent and parent.constrained():
            parent.enact_local_constraints()

    def unconstrained_duration(self):
        # Careful! Demands that all of the unconstrained beats sort themselves out
        return sum([u_b.duration() for u_b in self.unconstrained_beats])

    def duration(self, demand_predefined=False):
        if self.has_duration():
            return self._duration
        if demand_predefined:
            print "I must insist that this duration be defined previously."
            self.underconstrained_tree()

        # Reach up recursively for a duration; eventually we must hit one.
        parent = self.parent
        if not parent:
            print "I guess no assignment to", self.label + \
                    "'s subtree ever made it up this high."
            self.underconstrained_tree()
        if self.size:
            self._duration = (parent.duration() - parent.unconstrained_duration())* \
                    self.size
        else:
            if not self.beats:
                print "A leaf beat has no constraints. No way to figure that out..."
                self.underconstrained_tree()
            self.enact_local_constraints()
            if not self.has_duration():
                print "By the time we needed the duration of the unconstrained beat", \
                        self + ", it hadn't been defined fully."
                self.underconstrained_tree()
        return self._duration


    def time(self):
        if not self.parent:
            self._time = 0.0
        if self._time is not None:
            return self._time
        time = self.parent.time()
        for sibling_beat in self.parent.beats:
            if sibling_beat is self:
                break
            time += sibling_beat.duration()
        self._time = time
        return time

    def attach(self, sound, location):
        if sound is None:
            return #  Allow "None" to mean a skip in iterators
        self.sounds += [(sound, location)]

    def descendent_beats(self):
        if not self.beats:
            return [self]
        return [self] + list(chain(*[b.descendent_beats() for b in self.beats]))
    
    def __str__(self):
        return self.label + "@d=" + str(self._duration)

class Sound:

    ear_separation = .2
    standard_distance = .2 # the distance at which a sound is imagined to be heard when it's at unit volume
    c_sound = 344.0
    quick_play = False
    default_rate = 44100

    class CacheStatus:
        hits_before_cache = 2
        no_cache = 0
        smart_cache = 1
        instant_cache = 2

    def set_unique_rate(self, rate):
        try:
            if not self.rate:
                self.rate = rate
            elif self.rate != rate:
                print "Incompatible rates detected among composite sounds."
                exit()
        except:
            self.rate = rate

    def render_from(self, location, universal_cache={}):
        if self.cache_status == Sound.CacheStatus.no_cache:
            return self._render_from(location)
        tag = self.cache_tag + Sound.location_tag(location)
        if self.cache_status == Sound.CacheStatus.instant_cache:
            cached = universal_cache.get(tag, self._render_from(location))
            universal_cache[tag] = cached
            return cached
        if self.cache_status == Sound.CacheStatus.smart_cache:
            cached = universal_cache.get(tag, 0)
            if isinstance(cached, tuple):
                # Hit! Another sound similar enough to this one to replace it
                # was rendered here enough times in a row that it was cached.
                return cached 
            hits = cached + 1
            if hits == Sound.CacheStatus.hits_before_cache:
                rendered = self._render_from(location)
                universal_cache[tag] = rendered
                return rendered
            universal_cache[tag] = hits
            return self._render_from(location)
        else:
            print "Unknown cache status?? okaaay...?"
            exit()

    @staticmethod
    def location_tag(location):
        try:
            r, theta, phi = location
        except:
            # Only plane coordinates given... Assume phi = 0
            x, y = location
            r, theta, phi = sqrt(x*x + y*y), atan2(y, x), 0
        r_bin = int(log(r, 1.1))
        theta_bin = int(theta/pi*180/4) # 4-degree bins
        phi_bin = int(phi/pi*180/4)
        return '@' + str(r_bin) + '_' + str(theta_bin) + '_' + str(phi_bin)

    def duration(self):
        pass

    def _to_stereo(self, rate, mono_data, location):
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
        
        left_dist = sqrt((x + Sound.ear_separation / 2)**2 + y*y)
        right_dist = sqrt((x - Sound.ear_separation / 2)**2 + y*y)
        decays = np.array([[Sound.standard_distance/left_dist], [Sound.standard_distance/right_dist]])
        delays = np.array([[left_dist / Sound.c_sound], [right_dist / Sound.c_sound]])
        if Sound.quick_play:
            quick_data = np.hstack((np.zeros((int(delays.max() * rate) + 1,)), mono_data))
            return np.vstack((quick_data, quick_data)) * decays
        padded_data = np.hstack((mono_data, np.zeros((int(delays.max() * rate) + 1,))))
        # A shift of n is realized by a multiplication by exp(2pi*n*w/T) (but it can be fractional!)
        transform = np.tile(np.fft.rfft(padded_data), (2, 1))
        exp_coeff = -2j * pi * rate / len(padded_data)
        transformed = transform * np.exp(exp_coeff * delays * np.tile(np.arange(transform.shape[1]), (2, 1)))
        return np.fft.irfft(transformed) * decays

    def _read_mono_data(self, filename):
        filerate, data = wavfile.read(filename)
        return np.array(data).astype(np.float) / 2**15

    @staticmethod
    def sigmoid(samples):
        curve = lambda x: 1.0/(1.0 + np.exp(-x))
        return curve(np.arange(-samples/2, samples/2).astype(np.float)/samples * 2 * 5) #  5 "time constants" is nearly 1.0

class RawSound(Sound):

    def __init__(self, file_path, registration_point=None):
        if registration_point is not None:
            self.reg_pt = registration_point
        else:
            parse = os.path.basename(file_path).split('.')
            if len(parse) > 2:
                self.reg_pt = float(parse[1])/1000 # field after first dot in file name is reg_pt ms
            else:
                self.reg_pt = 0.0
        self._set_rate_from_file(file_path)
        self.file_path = file_path
        self._duration = None
        self.cache_status = Sound.CacheStatus.smart_cache
        self.cache_tag = "RawS\"" + self.file_path + "\"rg" + str(int(self.reg_pt*1000))

    def _set_rate_from_file(self, file_path):
        with contextlib.closing(wave.open(file_path,'r')) as f:
            self.rate = f.getframerate()

    def _render_from(self, location):
        mono_data = self._read_mono_data(self.file_path)
        return (self.reg_pt, self._to_stereo(self.rate, mono_data, location))

    def duration(self):
        if self._duration is None:
            with contextlib.closing(wave.open(self.file_path,'r')) as f:
                frames = f.getnframes()
                self._duration = frames / float(self.rate)
        return self._duration

class RandomSound(Sound):

    def __init__(self, sounds=None, cache=False):
        if sounds is None:
            self.sounds = []
        else:
            for sound in sounds:
                self.set_unique_rate(sound.rate)
            self.sounds = sounds
            self.update_cache_tag(self.sounds)
        self._duration = None
        if cache:
            self.cache_status = Sound.CacheStatus.instant_cache
        else:
            self.cache_status = Sound.CacheStatus.no_cache

    def _render_from(self, location):
        some_random_sound = self.sounds[np.random.randint(len(self.sounds))]
        return some_random_sound.render_from(location)

    def duration(self):
        if self._duration is None:
            self._duration = max([snd.duration() for snd in self.sounds])
        return self._duration

    def update_cache_tag(self, unique_sounds):
        self.cache_tag = "RandS" + str(hash(tuple(sorted([s.cache_tag for s in unique_sounds]))))

    def populate_with_dir(self, dir):
        for file_name in os.listdir(dir):
            parse = file_name.split('.')
            if parse[-1] != 'wav':
                continue
            new_raw_sound = RawSound(os.path.join(dir, file_name))
            self.set_unique_rate(new_raw_sound.rate)
            self.sounds.append(new_raw_sound)
        self.update_cache_tag(self.sounds)
        return self

class SpreadSound(Sound):

    def __init__(self, sound, x_spread, y_spread, t_spread,
            num_sounds, num_sounds_spread=0, cache=False):
        self.set_unique_rate(sound.rate)
        self.sound = sound
        self.x_spread = x_spread
        self.y_spread = y_spread
        self.t_spread = t_spread
        self.num_sounds = num_sounds
        self.num_sounds_spread = num_sounds_spread
        self.cache_tag = "SprSsnd(" + sound.cache_tag + ")xs" + str(int(x_spread*1000)) + \
                "ys" + str(int(y_spread*1000)) + "ts" + str(int(t_spread*1000)) + "no" + \
                str(num_sounds) + "ns" + str(num_sounds_spread)
        if cache:
            self.cache_status = Sound.CacheStatus.instant_cache
        else:
            self.cache_status = Sound.CacheStatus.no_cache

    def duration(self):
        return self.sound.duration() + self.t_spread * 3  # After 3 standard deviations, probability is acceptably low

    def _render_from(self, location):
        stereo_buffer = np.array([[],[]])
        if self.num_sounds_spread == 0:
            n = self.num_sounds
        else:
            n = max(int(np.random.normal(self.num_sounds, self.num_sounds_spread)), 0)
        reg_index = 0
        for s in range(n):
            center_x, center_y = location
            if self.x_spread == 0:
                x = center_x
            else:
                x = np.random.normal(center_x, self.x_spread) # TODO: Set seeds deterministically so that result for same call is the same
            if self.y_spread == 0:
                y = center_y
            else:
                y = np.random.normal(center_y, self.y_spread)
            sound_reg_pt, sound_data = self.sound.render_from((x, y))
            if self.t_spread == 0:
                t = 0
            else:
                t = np.random.normal(0, self.t_spread)
            start_index = reg_index + int((t - sound_reg_pt) * self.rate)
            if start_index < 0:
                start_index = abs(start_index)
                stereo_buffer = np.hstack((np.zeros((2, start_index)), stereo_buffer))
                reg_index += start_index
                start_index = 0
            end_index = start_index + sound_data.shape[1]
            if (end_index) >= stereo_buffer.shape[1]:
                stereo_buffer = np.hstack((stereo_buffer, np.zeros((2, end_index - stereo_buffer.shape[1] + 1))))
            stereo_buffer[:, start_index : end_index] = Track._mix(stereo_buffer[:, start_index : end_index], sound_data)
        return (float(reg_index)/self.rate, stereo_buffer)

class ClippedSound(Sound):

    def __init__(self, sound, clip_duration, offset=0.0, margin=.01, cache=False):
        self.set_unique_rate(sound.rate)
        self.sound = sound
        self.clip_duration = clip_duration
        self.margin = margin
        self.cap = Sound.sigmoid(int(margin * self.rate))
        if offset < 0 or offset > clip_duration:
            print "EEEW NEEW You can't initialize a Clipped Sound that might put the registration point off the sample."
            return
        self.offset = offset
        self.cache_tag = "ClSsnd(" + sound.cache_tag + "dr" + str(int(clip_duration*1000)) + \
                "os" + str(int(offset*1000))
        if cache:
            self.cache_status = Sound.CacheStatus.smart_cache
        else:
            self.cache_status = Sound.CacheStatus.no_cache

    def duration(self):
        return self.clip_duration + self.margin

    def _render_from(self, location):
        reg_pt, sound_data = self.sound.render_from(location)
        cap_offset = len(self.cap)/2
        start_index = int((reg_pt - self.offset) * self.rate - cap_offset)
        end_index = start_index + int(self.clip_duration * self.rate) + \
                cap_offset
        if start_index < 0:
            start_index = 0
            new_reg_pt = reg_pt
        else:
            new_reg_pt = self.offset
        if end_index > sound_data.shape[1]:
            end_index = sound_data.shape[1]
        clipped_data = sound_data[:, start_index : end_index]
        cap = self.cap[: clipped_data.shape[1]]
        clipped_data[:, : len(cap)] = clipped_data[:, : len(cap)] * cap
        clipped_data[:, -len(cap) :] = clipped_data[:, -len(cap) :] * \
                cap[::-1]
        return new_reg_pt, clipped_data


class RandomIntervalSound(Sound):

    def __init__(self, sound, interval=None, margin=.1, data=None, cache=False):
        self.set_unique_rate(sound.rate)
        self.sound = sound
        self.interval = interval
        self.margin = margin
        self.data = data
        self.cap = Sound.sigmoid(int(margin * self.rate))
        if interval:
            self.cache_tag = "RISsnd(" + sound.cache_tag + ")iv" + str(int(interval*1000))
            if cache:
                self.cache_status = Sound.CacheStatus.instant_cache
            else:
                self.cache_status = Sound.CacheStatus.no_cache

    def duration(self):
        return self.interval

    def _render_from(self, location):
        if self.data is None:
            self.data = self.sound.render_from((0, Sound.standard_distance))[1][0] #  ignore reg pt; take one track; remember it
        total_samples = len(self.data)
        samples = int(self.interval * self.rate)
        unclaimed_samples = samples
        supplementary_intervals = []
        cap_size = len(self.cap)
        while unclaimed_samples > total_samples/2:
            interval_size = np.random.randint(cap_size * 2, total_samples/2)
            unclaimed_samples -= (interval_size - cap_size)
            supplementary_intervals.append(self.random_data_of_length(
                interval_size))
        mono_data = self.random_data_of_length(unclaimed_samples)
        for subinterval in supplementary_intervals:
            mono_data = np.concatenate((mono_data[: -cap_size],
                mono_data[-cap_size :] + subinterval[: cap_size],
                subinterval[cap_size :]))
        return (0.0, self._to_stereo(self.rate, mono_data, location))

    def random_data_of_length(self, length):
        random_position = np.random.randint(len(self.data) - length)
        interval_data = np.array(self.data[random_position :
            random_position + length])
        eff_cap_size = min(len(self.cap), length)
        cap = self.cap[: eff_cap_size]
        interval_data[: eff_cap_size] = cap * interval_data[: eff_cap_size]
        interval_data[-eff_cap_size :] = cap[::-1] * \
                interval_data[-eff_cap_size :]
        return interval_data

    def for_interval(self, interval):
        return RandomIntervalSound(self.sound, interval, self.margin, self.data)

class ResampledSound(Sound):

    def __init__(self, sound, freq_func, cache=True):
        self.set_unique_rate(sound.rate)
        self.sound = sound
        self.freq_func = np.vectorize(freq_func)
        self._duration = None
        if cache:
            self.cache_status = Sound.CacheStatus.smart_cache
        else:
            self.cache_status = Sound.CacheStatus.no_cache
        self.cache_tag = "ResSsnd(" + sound.cache_tag + "),func" + \
            str(hash(freq_func.__code__.co_code)) + "iv" + str(int(1000000*freq_func(0)))

    def duration(self):
        if self._duration is None:
            sound_duration = self.sound.duration()
            self._duration = np.trapz(1.0/self.freq_func(np.arange(-sound_duration*self.rate, sound_duration*self.rate)))/self.rate
        return self._duration

    def _render_from(self, location):
        block_size = 10000
        reg_pt, sound_data = self.sound.render_from(location)
        index_counter = np.arange(sound_data.shape[1])
        resampled_data = np.array([[],[]])
        end_marker = 2 # outside range of audio data
        # negative times
        i = -1
        stop = False
        mark = reg_pt * self.rate
        while not stop:
            interp_offsets, next_pt = self._points_for_block(block_size, i)
            interp_points = interp_offsets - next_pt + mark # i_off[-1] is the maximum
            block_left = np.interp(interp_points, index_counter, sound_data[0], left=end_marker)
            block_right = np.interp(interp_points, index_counter, sound_data[1], left=end_marker)
            if block_left[0] == end_marker:
                block_left = block_left[block_left != end_marker]
                if len(block_left) > 0:
                    block_right = block_right[-block_left.size:]
                else:
                    block_right = np.array([])
                stop = True
            block = np.vstack((block_left, block_right))
            resampled_data = np.hstack((block, resampled_data))
            mark = interp_points[0]
            i -= 1

        new_reg_pt = float(resampled_data.shape[1])/self.rate

        # positive times
        i = 0
        stop = False
        mark = reg_pt * self.rate
        while not stop:
            interp_offsets, next_pt = self._points_for_block(block_size, i)
            interp_points = interp_offsets + mark
            block_left = np.interp(interp_points, index_counter, sound_data[0], right=end_marker)
            block_right = np.interp(interp_points, index_counter, sound_data[1], right=end_marker)
            if block_left[-1] == end_marker:
                block_left = block_left[block_left != end_marker]
                block_right = block_right[:block_left.size]
                stop = True
            block = np.vstack((block_left, block_right))
            resampled_data = np.hstack((resampled_data, block))
            mark = interp_points[0] + next_pt
            i += 1
        return (new_reg_pt, resampled_data)

    def _points_for_block(self, block_size, block_number):
        block_start_index = block_size * block_number
        block_end_index = block_start_index + block_size
        intervals = self.freq_func(np.arange(block_start_index, block_end_index).astype(np.float)/self.rate)
        summed = np.cumsum(intervals)
        resample_points = np.hstack(([0], summed[:-1]))
        next_point = summed[-1]
        return (resample_points, next_point)

class PitchedSound(Sound):

    chromatic_scale = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    note_map = {name: index for index, name in enumerate(chromatic_scale)}
    note_map.update([("Bb", 1), ("Db", 4), ("Eb", 6), ("Gb", 9), ("Ab", 11)])
    temper_ratio = 2.0**(1.0/12)

    def for_pitch(self, pitch):
        frequency = PitchedSound.resolve_pitch(pitch)
        ratio = frequency/PitchedSound.resolve_pitch(self.pitch)
        if abs(ratio - 1.0) < .00001:
            return self
        return ResampledSound(self, (lambda x, r=ratio: r))

    @staticmethod
    def resolve_pitch(pitch):
        try:
            return float(pitch)
        except:
            note_name, octave_str = pitch.split('_')
            return PitchedSound.note_frequency(note_name, int(octave_str))
    
    @staticmethod
    def note_frequency(name, octave=4):
        scale_index = PitchedSound.note_map[name]
        return 440 * PitchedSound.temper_ratio**scale_index * 2**(octave - 4 - (scale_index % 12 > 2))

class RawPitchedSound(RawSound, PitchedSound):

    def __init__(self, file_path, registration_point=None, pitch=None):
        self._set_rate_from_file(file_path)
        parse = os.path.basename(file_path).split('.')
        if registration_point is not None:
            self.reg_pt = registration_point
        else:
            try:
                self.reg_pt = float(parse[1])/1000 # field after first dot in file name is reg_pt ms
            except:
                self.reg_pt = 0.0
        if pitch is None:
            pitch = parse[2] # field after second dot is frequency, Hz
        self.pitch = pitch
        self.file_path = file_path
        self._duration = None
        self.cache_status = Sound.CacheStatus.smart_cache
        self.cache_tag = "RawPS\"" + self.file_path + "\"rg" + str(int(self.reg_pt*1000)) + \
                "@p" + str(int(PitchedSound.resolve_pitch(pitch)*1000))


class RandomPitchedSound(RandomSound, PitchedSound):

    def __init__(self, pitch=None, pitched_sounds=None, cache=False):
        if pitched_sounds is None:
            self.pitched_sounds = []
            self.rate = None
        else:
            for sound in pitched_sounds:
                self.set_unique_rate(sound.rate)
            self.pitched_sounds = pitched_sounds[:]
            self.update_cache_tag(self.pitched_sounds)
        self.pitch = pitch  # if it's None, it's expected that we'll call
                            # for_pitch later
        if pitch is None:
            self.sounds = []
        else:
            self.sounds = [pitched_sound.for_pitch(self.pitch) for \
                    pitched_sound in self.pitched_sounds]
        self._duration = None
        if cache:
            self.cache_status = Sound.CacheStatus.instant_cache
        else:
            self.cache_status = Sound.CacheStatus.no_cache

    def populate_with_dir(self, dir):
        for file_name in os.listdir(dir):
            parse = file_name.split('.')
            if parse[-1] != "wav":
                continue
            new_pitched_sound = RawPitchedSound(os.path.join(dir, file_name))
            self.set_unique_rate(new_pitched_sound.rate)
            self.pitched_sounds.append(new_pitched_sound)
            if self.pitch is not None:
                self.sounds.append(new_pitched_sound.for_pitch(self.pitch))
        self.update_cache_tag(self.pitched_sounds)
        return self

    def for_pitch(self, pitch): #  override with something smarter than just resampling
        return RandomPitchedSound(pitch, self.pitched_sounds)

class Instrument:

    def __init__(self, data_dir, filter=[]):
        self.note_dict = {}
        for note_file_name in os.listdir(data_dir):
            parse = note_file_name.split('.')
            key = parse[0]
            if not filter != [] and key not in filter:
                continue
            if len(parse) > 2:
                reg_pt = float(parse[1])
            else:
                reg_pt = 0.0
            new_entry = (reg_pt, data_dir + '/' + note_file_name)
            try:
                self.note_dict[key] = self.note_dict[key] + [new_entry]
            except KeyError:
                self.note_dict[key] = [new_entry]
            self.notelist.append(new_entry)

    def gen_note_file(self, key=None):
        if key is None:
            notelist = self.notelist
        else:
            notelist = self.note_dict[key]
        return notelist[numpy.random.randint(len(notelist))]

    def __getitem__(self, filter):
        return Instrument(self.data_dir, filter)

class Track:

    def __init__(self, name, rate, duration=None, start_time=None, volume=1.0,
            padding=.1, end_padding=None):
        self.name = name
        self.rate = rate
        self.top_beat = Beat(label=(name + "_top"))
        if duration:
            self.top_beat.set_duration(duration)
        self.time_lending_track = self.time_lending_beat = None
        self._start_time = start_time
        self.volume = volume
        self.reg_index = 0
        self.pre_padding_samples = padding * self.rate
        if end_padding is None:
            self.post_padding_samples = self.pre_padding_samples
        else:
            self.post_padding_samples = end_padding * self.rate
        self._data = None

    def data(self):
        # finally decide that we need a concrete duration and data length
        if self._data is not None:
            return self._data
        self.total_samples = self.pre_padding_samples + int(self.rate * \
                self.duration()) + self.post_padding_samples + 1
        self._data = np.zeros((2, self.total_samples))
        return self._data

    def mix_into(self, t0, buffer): # t0 is mixer time, not track time.
        self.data()
        track_start_index = int((t0 - self.start_time()) * self.rate) + \
                self.pre_padding_samples
        buffer_length = buffer.shape[1]
        if track_start_index < 0:
            buffer_write_index = -track_start_index
            if buffer_write_index > buffer_length:
                return buffer
            track_start_index = 0
            print "Be warned, ya dope! You outstayed your welcome in the " \
                    "pre-padding for the", self.name, "track" 
        else:
            buffer_write_index = 0
        track_end_index = track_start_index - buffer_write_index + \
                buffer_length
        buffer_end_write_index = buffer_length
        if track_end_index > self.total_samples:
            buffer_end_write_index -= track_end_index - self.total_samples
            if buffer_end_write_index < 0:
                return buffer
            track_end_index = self.total_samples
            print "Wowzas warning mah brudda, you've overtaxed track " + \
                    self.name + "'s post-padding"
        buffer_data = buffer[:, buffer_write_index : buffer_end_write_index]
        self._render(track_start_index, track_end_index)
        track_data = self.data()[:, track_start_index : track_end_index] * \
                self.volume
        buffer[:, buffer_write_index : buffer_end_write_index] = Track._mix(
                track_data, buffer_data)
        return buffer

    def link_root(self, beat, time_lending_track=None):
        if time_lending_track:
            test_beat = beat
            while test_beat.parent:
                test_beat = test_beat.parent
            if test_beat is not time_lending_track.top_beat:
                print "Yayaya no, the beat provided to link " \
                    "isn't even in the track."
            self.time_lending_track = time_lending_track
        self.top_beat.link(beat)
        self.time_lending_beat = beat
        return self.top_beat

    @staticmethod
    def _mix(a, b):
        return a + b

    def duration(self):
        return self.top_beat.duration()

    def start_time(self):
        if self._start_time is not None:
            return self._start_time
        if self.time_lending_track:
            self._start_time = self.time_lending_beat.time() + \
                    self.time_lending_track.start_time()
        else:
            self._start_time = self.time_lending_beat.time()
        return self._start_time

    def _render(self, track_start_index, track_end_index):
        t0 = (track_start_index - self.pre_padding_samples)/self.rate
        t1 = (track_end_index - self.pre_padding_samples)/self.rate
        sounds_rendered = 0
        for beat in self.top_beat.descendent_beats():
            for sound, location in beat.sounds:
                if not (beat.time() + sound.duration() > t0 and \
                        beat.time() - sound.duration() < t1):
                    continue
                sounds_rendered += 1
                reg_pt, sound_data = sound.render_from(location)
                sound_start_time = beat.time() - reg_pt
                start_index = int(sound_start_time * self.rate) + \
                        self.pre_padding_samples
                if start_index < 0:
                    start_index = 0
                    sound_data = sound_data[:, -start_index:]
                end_index = start_index + sound_data.shape[1]
                if end_index > self.total_samples:
                    end_index = self.total_samples
                    sound_data = sound_data[:, : end_index - start_index]
                self.data()[:, start_index : end_index] = \
                        Track._mix(self.data()[:, start_index : end_index], \
                        sound_data)
        print "Rendered", sounds_rendered, "sounds for track", self.name

    def __str__(self):
        return self.name + (" (%.1f-%.1f)" % (self._start_time, self._start_time + \
                self.duration()) if self.top_beat.has_duration() and self._start_time \
                else " (no times)")



class Mixer:

    attenuation_boost = 2

    def __init__(self, name, rate=Sound.default_rate, tracks=[]):
        self.name = name
        self.tracks = tracks
        self.rate = rate

    def set_rate(self, rate):
        self.rate = rate

    def _pyaudio_play(self, wavname):
        wf = wave.open(wavname, 'rb')
        if wf.getframerate() != self.rate:
            print "Don't play back a work with the wrong rate!"
            exit()
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                channels=wf.getnchannels(),
                                rate=wf.getframerate(),
                                output=True)
        chunksize = 1024
        data = wf.readframes(chunksize)
        while data != '':
            stream.write(data)
            data = wf.readframes(chunksize)
        stream.stop_stream()
        stream.close()
        p.terminate()

    def play(self, t0=0, t1=None, quick_play=True):
        self.render_to_file('temp.wav', t0, t1, quick_play=quick_play)
        print "Begin playback."
        self._pyaudio_play('temp.wav')

    def play_beat(self, beat, quick_play=True):
        self.play(beat.time(), beat.time() + beat.duration(), quick_play)

    @staticmethod
    def play_sound(sound, location, number=1, quick_play=True):
        Sound.quick_play = quick_play
        buffer = np.array([[],[]])
        for n in range(number):
            buffer = np.hstack((buffer, sound.render_from(location)[1]))
        Mixer.write_to_file(sound.rate, "sound_temp.wav", buffer)
        self._pyaudio_play('sound_temp.wav')

    def render_to_file(self, out_file_name, t0=None, t1=None, quick_play=False):
        check_file_free = open(out_file_name, 'w').close()
        Sound.quick_play = quick_play
        if not t0:
            t0 = min([track.start_time() for track in self.tracks])
        if not t1:
            t1 = max([track.start_time() + track.duration() for track in self.tracks])
        data_buffer = np.zeros((2, int(self.rate * (t1 - t0)) + 1))
        for track in self.tracks:
            data_buffer = track.mix_into(t0, data_buffer)
        Mixer.write_to_file(self.rate, out_file_name, data_buffer)

    @staticmethod
    def write_to_file(rate, out_file_name, buffer):
        data_buffer = (buffer * 2**15 * Mixer.attenuation_boost).astype(np.int16)
        print "Finished rendering, writing out buffer..."
        print "Max level:"
        print float(data_buffer.max())/2**15
        output_wav = wave.open(out_file_name, 'w')
        output_wav.setparams((2, 2, rate, 0, 'NONE', 'not compressed')) # (nchannels, samplewidth, framerate, nframes, compressiontype, compressionname)
        write_chunk_size = 10000
        write_chunk = ''
        for left_sample, right_sample in zip(data_buffer[0], data_buffer[1]):
            left_bytes = struct.pack('h', left_sample)
            right_bytes = struct.pack('h', right_sample)
            write_chunk += ''.join((left_bytes, right_bytes))
            if len(write_chunk) == write_chunk_size:
                output_wav.writeframes(write_chunk)
                write_chunk = ''
        output_wav.writeframes(write_chunk)
        output_wav.close()
