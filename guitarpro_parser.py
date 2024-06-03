from guitarpro import parse
import numpy as np

song = parse('ALONE.gp5')

bpm = 116.6 # part 1
# bpm = 116.5  # part 2
# bpm = song.tempo
if song.tracks[0].measures[0].timeSignature.numerator != 4 \
        or song.tracks[0].measures[0].timeSignature.denominator.value != 4:
    raise NotImplementedError

# a whole note should last 4*60/bpm sec
durations = []  # [duration_sec]
for i, measure in enumerate(song.tracks[0].measures):
    for beat in measure.voices[0].beats:
        if beat.status.value == 0:  # empty measure
            durations.append(4 * 60 / bpm)
            continue

        _beat_duration_sec = 1/bpm/16 * beat.duration.time
        durations.append(_beat_duration_sec)

        # check for not implemented features
        for note in beat.notes:
            if note.effect.slides:
                if note.effect.slides[0].value != 2:
                    raise NotImplementedError

durations = np.array(durations)

np.save('durations.npy', durations)