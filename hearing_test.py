from musp import *
from math import pi, sin, cos
from itertools import product, repeat
from mpl_toolkits.mplot3d import Axes3D

delay = .5
batch_interval = 1
batch_num = 2

datadir = os.path.expanduser("~/.mu-sp")

def aulib(sound_dir):
    return os.path.join(datadir, "audio", sound_dir)

cube_size = 4
#test_locs = [Location(x, y, z) for z, x, y in product([-cube_size, cube_size], repeat=3)]
test_locs = [Location(a/200.0, 5*sin(b*2*pi/30), 5*cos(b*2*pi/30)) for a, b in zip(range(-50, 50), range(100))]
plot_pts = []
for l in test_locs:
    print l
    plot_pts.append([c for c in l])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(*zip(*plot_pts))
plt.show()

test_sound = RandomSound()
test_sound.populate_with_dir(aulib("audiofile"))

root = Beat()
test_track = Track("tests!", Sound.default_rate, padding=.5, end_padding=2)
for batch_beat, loc in zip(test_track.link_root(root).split(len(test_locs)), test_locs):
    batch_beat.set_duration(batch_interval)
    test_beat = batch_beat.split([1, None])[0]
    for each_sound_beat in test_beat.split(batch_num):
        each_sound_beat.set_duration(delay)
        each_sound_beat.attach(test_sound, loc)

mix = Mixer("Let's make some art, I guess...!", Sound.default_rate, [test_track])
mix.play(quick_play=False)
