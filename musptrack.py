from musp import Beat
import numpy as np

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

