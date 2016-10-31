from musp import Track, Location, DiscreteEarDelayAS, discretized_delay_as
import os, contextlib, wave
import numpy as np
from math import sqrt, atan2, log, pi
import scipy.io.wavfile as wavfile

class Sound:

    quick_play = False
    default_rate = 44100
    default_aural_space = discretized_delay_as(default_rate)
                    # DiscreteEarDelayAS("default_discrete_ear_delay_as", default_rate)

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
        tag = self.cache_tag + Location(location).cache_tag()
        if self.cache_status == Sound.CacheStatus.instant_cache:
            cached = universal_cache.get(tag, self._render_from(location))
            universal_cache[tag] = cached
            return cached
        if self.cache_status == Sound.CacheStatus.smart_cache:
            cached = universal_cache.get(tag, 0)
            if isinstance(cached, tuple):
                # Hit! Another sound similar enough to this one to replace it
                # was rendered here enough times in a row that it was cached.
                print "got a cache!"
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

    def duration(self):
        pass

    def _to_stereo(self, rate, mono_data, location, astf=None):
        if Sound.quick_play:
            decays = np.array(zip(Location(location).decays_at_ears()))
            delays = np.array(zip(Location(location).delays_to_ears()))
            quick_data = np.hstack((np.zeros((int(delays.max() * rate) + 1,)), mono_data))
            return np.vstack((quick_data, quick_data)) * decays
        if astf is None:
            astf = self.default_aural_space.astf_for_location(location)
        astf_data, impulse_response_length = astf.generate_astf()
        print astf
        print id(astf_data)
        padded_data = np.hstack((mono_data, np.zeros((1,))))
        transform = np.tile(np.fft.rfft(padded_data[:astf_data.shape[1]*2 - 1]), (2, 1))
        transformed = transform * astf_data
        transformed_back = np.fft.irfft(transformed)
        return np.hstack((transformed_back, np.zeros((2, len(mono_data) - transformed_back.shape[1]))))

    def set_default_aural_space(self, aural_space):
        self.default_aural_space = aural_space

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
        self.default_aural_space = Sound.default_aural_space
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
        self.default_aural_space = Sound.default_aural_space
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

    def __init__(self, sound, spread_vector, t_spread,
            num_sounds, num_sounds_spread=0, cache=False):
        self.set_unique_rate(sound.rate)
        self.sound = sound
        if min(spread_vector) < 0:
            print "Can't use a spread vector with negative components. Keep it Quad 1 yall."
            exit()
        self.spread_vector = spread_vector
        self.t_spread = t_spread
        self.num_sounds = num_sounds
        self.num_sounds_spread = num_sounds_spread
        self.default_aural_space = Sound.default_aural_space
        self.cache_tag = "SprSnd(" + sound.cache_tag + ")" + \
                "sv" + str([int(coord*1000) for coord in self.spread_vector]) + \
                "ts" + str(int(t_spread*1000)) + "no" + str(num_sounds) + \
                "ns" + str(num_sounds_spread)
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
            jittered_location = Location((coord if coord_spread == 0 else \
                    np.random.normal(coord, coord_spread) for coord, coord_spread in \
                    zip(location, self.spread_vector)))
            sound_reg_pt, sound_data = self.sound.render_from(jittered_location)
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
                stereo_buffer = np.hstack((stereo_buffer, np.zeros(
                    (2, end_index - stereo_buffer.shape[1] + 1))))
            stereo_buffer[:, start_index : end_index] = Track._mix(
                    stereo_buffer[:, start_index : end_index], sound_data)
        return (float(reg_index)/self.rate, stereo_buffer)

class ClippedSound(Sound):

    def __init__(self, sound, clip_duration, offset=0.0, margin=.01, cache=False):
        self.set_unique_rate(sound.rate)
        self.sound = sound
        self.clip_duration = clip_duration
        self.margin = margin
        self.cap = Sound.sigmoid(int(margin * self.rate))
        self.default_aural_space = Sound.default_aural_space
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
        self.default_aural_space = Sound.default_aural_space
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
            self.data = self.sound.render_from((0, Location.standard_distance, 0))[1][0] #  ignore reg pt; take one track; remember it
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
        self.default_aural_space = Sound.default_aural_space
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
        self.default_aural_space = Sound.default_aural_space
        self.cache_status = Sound.CacheStatus.no_cache#smart_cache
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
        self.default_aural_space = Sound.default_aural_space
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
