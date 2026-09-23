"""Parse the reference transcription into plain note tuples (beats)."""
import mido

REF = __file__.rsplit('/', 2)[0] + '/ref/touhou_7_phantasm_boss_yukari.mid'
PICKUP = 1.0  # song has a 1-beat 1/4 pickup bar before bar 0


def load():
    m = mido.MidiFile(REF)
    tpb = m.ticks_per_beat
    tracks = {}
    for i, t in enumerate(m.tracks):
        tick = 0
        on = {}
        notes = []
        for x in t:
            tick += x.time
            if x.type == 'note_on' and x.velocity > 0:
                on[(x.channel, x.note)] = (tick, x.velocity)
            elif x.type in ('note_off', 'note_on'):
                k = (x.channel, x.note)
                if k in on:
                    s, v = on.pop(k)
                    # beat 0 = downbeat of bar 0 (after the pickup)
                    notes.append((s / tpb - PICKUP, (tick - s) / tpb, x.note, v / 127))
        if notes:
            tracks[i] = sorted(notes)
    return tracks
