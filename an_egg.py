from proto import *
from math import pi, sin, cos
import numpy as np

tonic_freq = PitchedSound.note_frequency("F#", 3)
tempo_dur = 2.6/7

def aulib(sound_dir):
    return os.path.join("audio", sound_dir)

def rhlib(rh_name):
    return os.path.join("an_egg_rh", rh_name + ".rh")

def loctrans(far, angle):
    return (far*sin(angle), far*cos(angle))

def halftones_for_scale_deg(degree):
    semitones = [0, 2, 3, 5, 7, 8, 10][int(degree) - 1]
    if degree % 1 == .5:
        semitones += 1 
    return semitones

def deg_freq(degree):
    octave_mult = 1
    while degree > 7:
        degree -= 7
        octave_mult *= 2
    return tonic_freq*octave_mult * PitchedSound.temper_ratio**halftones_for_scale_deg(degree)

def fundamental_rhythm(beat):
    return beat.split([3, 3, 7, 3])

def apply_rhythm(beat, rhythm_file, key_sound_map):
    with open(rhythm_file) as rf:
        char_times = eval(''.join(rf.readline()))
    beat_map = beat.interleave_split(char_times)
    for key, beats in beat_map.iteritems():
        for beat in beats:
            try:
                for sound, loc in key_sound_map[key]:
                    beat.attach(sound, loc)
            except:
                beat.attach(*key_sound_map[key])

crystal_sound = RandomPitchedSound()
crystal_sound.populate_with_dir(aulib("crystal_ding"))

def add_tracks_fromto(tracklist, listoftracks):
    for track in tracklist:
        listoftracks.append(track)

def crystal_sounding(beat):
    beat.set_duration(tempo_dur*7)
    part1 = "565765243"
    crys1 = Track("crystals...", Sound.default_rate, padding=.5, end_padding=2)
    crys1_root = crys1.link_root(beat)
    with open(rhlib("crys_1_j")) as rf:
        char_times = eval(''.join(rf.readline()))
    beats = crys1_root.interleave_split(char_times)['j']
    
    for b, deg in zip(beats, part1):
        b.attach(crystal_sound.for_pitch(4*deg_freq(int(deg))), loctrans(4, pi/2))

    return [crys1]

def crystal_rise(beat):
    beat.set_duration(tempo_dur*7)
    part2 = "34567"
    crys2 = Track("more crystals...", Sound.default_rate, padding=.5, end_padding=2)
    crys2_root = crys2.link_root(beat)
    beats = crys2_root.split_even(14)[9:]
    
    for b, deg in zip(beats, part2):
        b.attach(crystal_sound.for_pitch(2*deg_freq(int(deg))), loctrans(4, -pi/2))

    return [crys2]

def crystal_complex(beat):
    beat.set_duration(tempo_dur*14)
    #part3 = "78765774554"*2
    part3 = "17876577547187657232"
    crys3 = Track("more (muy complicated) crystals...", Sound.default_rate, padding=.5, end_padding=2)
    crys3_root = crys3.link_root(beat)
    beats = crys3_root.split([1, .5, .5, 1, 1, 1, 2, #groups of 7
                              2, 1, 2, 2,
                              2, 2, 1, 1, 1,
                              3, 1, 1, 2])
    
    for b, deg in zip(beats, part3):
        deg = int(deg) + 8
        b.attach(crystal_sound.for_pitch(deg_freq(int(deg)+4)), loctrans(4, -pi/6))

    return [crys3]

def apply_each_half(beat, one_beat_function, firsthalf=True, secondhalf=True):
    if not firsthalf and not secondhalf:
        return
    sometracks = []
    try:
        b1, b2 = beat.beats
    except:
        b1, b2 = beat.split_even(2)
    if firsthalf:
        add_tracks_fromto(one_beat_function(b1), sometracks)
    if secondhalf:
        add_tracks_fromto(one_beat_function(b2), sometracks)
    return sometracks

def crystal_compiled_block(beat, levels):
    level_funcs = [lambda b: apply_each_half(b, crystal_sounding, True, False),
                   lambda b: apply_each_half(b, crystal_sounding, False, True),
                   lambda b: apply_each_half(b, crystal_rise, False, True),
                   lambda b: apply_each_half(b, crystal_rise, True, False),
                   crystal_complex]
    allthesetracks = []
    for l in levels:
        add_tracks_fromto(level_funcs[l](beat), allthesetracks)
    return allthesetracks

bow_violin_sound = RandomPitchedSound()
bow_violin_sound.populate_with_dir(aulib("bowed_violin"))

pluck_violin_sound = RandomPitchedSound()
pluck_violin_sound.populate_with_dir(aulib("plucked_violin_ring"))
pluck_violin_sound.populate_with_dir(aulib("plucked_violin_damp"))

def vibrato_snd_for_beat_frac(beat, deg, f, distance, sound=bow_violin_sound, h=0):
    # h is vibrato hertz
    vibrato_f = lambda t: PitchedSound.temper_ratio**(.25/(1.0 + np.exp(-t * 3))*sin(t*h*(2*pi)))
    beat.attach(ClippedSound(ResampledSound(sound.for_pitch(deg_freq(float(deg))), vibrato_f, cache=False), tempo_dur*f),
            loctrans(distance, -pi/3))

def violin_pluck_chords(beat):
    violin1 = Track("Violin me once!", Sound.default_rate)
    violin_root = violin1.link_root(beat)

    degrees = (1, 1, 1, 1, 1, 1)
    durations = (1, 1, 2, 3, 5, 2)
    distances = (4, 3, 2, 4, 2, 3)
    for deg, dur, dist, b in zip(degrees, durations, distances, violin_root.split(durations)):
        vibrato_snd_for_beat_frac(b, deg, dur, dist/5.0, sound=pluck_violin_sound, h=7)

    violin2 = Track("Violin me twice!", Sound.default_rate)
    violin_root = violin2.link_root(beat)

    degrees = (5, 5, 5, 4, 4, 3)
    durations = [d + .05 for d in (1, 1, 2, 3, 5, 2)]
    distances = (3, 3.5, 3, 2, 2, 4)
    for deg, dur, dist, b in zip(degrees, durations, distances, violin_root.split(durations)):
        vibrato_snd_for_beat_frac(b, deg, dur + .1, dist/5.0, sound=pluck_violin_sound, h=7)

    violin3 = Track("Violin me thrice!", Sound.default_rate)
    violin_root = violin3.link_root(beat)

    degrees = (7, 6, 7, 7, 6, 4)
    durations = [d - .05 for d in (1, 1, 2, 3, 5, 2)]
    distances = (4, 3.5, 4, 3, 4, 3.5)
    for deg, dur, dist, b in zip(degrees, durations, distances, violin_root.split(durations)):
        vibrato_snd_for_beat_frac(b, deg, dur + .1, dist/5.0, sound=pluck_violin_sound, h=7)

    return [violin1, violin2, violin3]

werb_raw = RawPitchedSound(os.path.join(aulib("werb_sine"), "werb_sine.0.110.wav"))
werb_sounds = {}

def werb_for_beat_frac(beat, degree, duration, distance):
    if degree not in werb_sounds:
        werb_sounds[degree] = RandomIntervalSound(werb_raw.for_pitch(.49*deg_freq(degree)), margin=.01)
    werb_sound = werb_sounds[degree]
    beat.attach(werb_sound.for_interval(duration*tempo_dur), loctrans(distance, pi))

def werb_under(beat):
    werb = Track("werbtrack", Sound.default_rate)
    werb_root = werb.link_root(beat)
    for b, d in zip(werb_root.split_even(4), (1, 2, 3, 4)):
        werb_for_beat_frac(b, d, 14.0/4, .5)
    return [werb]

random_mid_drum = RandomSound()
random_mid_drum.populate_with_dir(aulib("snares_off"))
mid_drum = SpreadSound(random_mid_drum, .2, .2, 0, 1)

def descending_snaresoff_tuple(beat, n):
    beats = [beat] if n is 1 else beat.split_even(n)
    for b, i in zip(beats, range(n, 0, -1)):
        b.attach(mid_drum, loctrans(i + .2, pi*10/3*i))

def mid_drum_rhythm(beat):
    drum = Track("Snares off please", Sound.default_rate)
    drum_root = drum.link_root(beat)
    one, two, three, four, five, six, seven = drum_root.split_even(7)
    descending_snaresoff_tuple(one, 2)
    descending_snaresoff_tuple(two, 1)
    descending_snaresoff_tuple(three, 3)
    descending_snaresoff_tuple(four, 4)
    descending_snaresoff_tuple(five, 1)
    descending_snaresoff_tuple(six, 6)
    descending_snaresoff_tuple(seven, 1)
    return [drum]

def create_main(beat):
    trackbag = []
    for levels, crystaltest in zip([(0, 1), (0, 1, 2), (0, 1, 2, 4), (0, 1, 2, 3, 4), (2, 3, 4)], beat.split(5)):
        add_tracks_fromto(crystal_compiled_block(crystaltest, levels), trackbag)
        add_tracks_fromto(violin_pluck_chords(crystaltest), trackbag)
        add_tracks_fromto(werb_under(crystaltest), trackbag)
        add_tracks_fromto(apply_each_half(crystaltest, mid_drum_rhythm), trackbag)

    return trackbag

mainbeat = Beat()
mix = Mixer("Let's make some art, I guess...!", Sound.default_rate, create_main(mainbeat))
mix.play()

