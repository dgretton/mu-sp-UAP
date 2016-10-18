import wave, struct, pyaudio
from muspsound import Sound
import numpy as np


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
