import random
from musp import *
from math import pi, sin, cos
from itertools import product, repeat
from mpl_toolkits.mplot3d import Axes3D

delay = .5
batch_interval = 2
batch_num = 7

datadir = os.path.expanduser("~/.mu-sp")

def aulib(sound_dir):
    return os.path.join(datadir, "audio", sound_dir)

cube_size = 4
test_locs = list(enumerate([Location(x, y, z) for z, x, y in product([-cube_size, cube_size], repeat=3)]))
#test_locs = list(enumerate(Location(0, -3, 0) for i in range(5)))
random.shuffle(test_locs)
plot_pts = []
print "test point order:"
for i, l in test_locs:
    print i
    plot_pts.append([c for c in l])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*zip(*plot_pts))

source_sound = RandomSound()
source_sound.populate_with_dir(aulib("crescendo_violin"))
test_sound = SpreadSound(source_sound, [cube_size/6.0]*3, 0, 1)

root = Beat()
test_track = Track("tests!", Sound.default_rate, padding=.5, end_padding=2)
for batch_beat, (loc_num, loc) in zip(test_track.link_root(root).split(len(test_locs)), test_locs):
    batch_beat.set_duration(batch_interval)
    print loc_num
    test_beat = batch_beat.split([1, None])[0]
    for each_sound_beat in test_beat.split(batch_num):
        each_sound_beat.set_duration(delay)
        each_sound_beat.attach(test_sound, loc)

mix = Mixer("Let's make some art, I guess...!", Sound.default_rate, [test_track])
mix.play(quick_play=False)
