import Tkinter as tk
import time, string, sys

try:
    out_file = sys.argv[1]
except:
    print "Provide name of file in which to save results as first argument."
    exit()

root = tk.Tk()
recording = False
loop_ms = None
loop_num = 0
start_time = None
rhythms = {}
after_ids = []

def kill_afters():
    for id in after_ids:
        root.after_cancel(id)

def onKeyPress(event):
    global start_time, recording, loop_ms
    
    t = time.time()
    c = event.char

    if not start_time:
        start_time = time.time()

    t_ms  = int((t - start_time) * 1000)
    in_loop_ms = t_ms % loop_ms if loop_ms else t_ms

    if c == '\r' and not loop_ms and recording:
        loop_ms = t_ms
        schedule_sounds()

    if c == '' or c not in string.lowercase:
        return
    if c not in rhythms:
        rhythms[c] = [in_loop_ms]
    else:
        rhythms[c].append(in_loop_ms)

    recording = True

def schedule_sounds():
    global loop_num, after_ids
    t_ms  = int((time.time() - start_time) * 1000)
    current_loop_number = t_ms/loop_ms
    if current_loop_number == loop_num:
        return

    # loop number changed
    loop_num = current_loop_number
    kill_afters()
    in_loop_ms = t_ms % loop_ms

    delay = loop_ms - in_loop_ms + 1
    if delay > 100:
        after_ids = [root.after(delay, schedule_sounds),
                root.after(delay - 1, root.bell)]

    for r in rhythms:
        for note_time_ms in rhythms[r]:
            delay = note_time_ms - in_loop_ms
            if delay > 0:
                after_ids.append(root.after(delay, root.bell))
        
root.bind('<KeyPress>', onKeyPress)
root.mainloop()

kill_afters()

with open(out_file, 'w') as f:
    for r in rhythms:
        times = rhythms[r]
        times.sort()
        rhythms[r] = [float(t)/loop_ms for t in times]
    #     rhythms[r] = [t2 - t1 for t1, t2 in zip(pre_r, pre_r[1:])]
    #     while 0 in rhythms[r]:
    #         rhythms[r].remove(0)
        print r, "==>", rhythms[r]
    f.write(repr(rhythms))
    f.write('\n' + repr(loop_ms))

