from itertools import chain

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

