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

            # if note.effect.grace:
            #     if not note.effect.grace.isOnBeat:
            #         raise NotImplementedError

# notes = []  # [duration_sec notetype string value isSlide isBend]
# for i, measure in enumerate(song.tracks[0].measures):
#     for beat in measure.voices[0].beats:
#         if beat.status.value == 0:  # empty measure
#             notes.append([4 * 60 / bpm, 0, 0, 0, 0, 0])
#             continue
#
#         _beat_duration_sec = 1/bpm/16 * beat.duration.time
#         if beat.status.value == 2:  # rest
#             notes.append([_beat_duration_sec, 0, 0, 0, 0, 0])
#
#         # playing
#         for note in beat.notes:
#             if note.value == 0:
#                 if note.type.value == 2:
#                     # tied notes can also bend
#                     notes.append([_beat_duration_sec, 2, note.string, notes[-1][3], 0, 0])
#                 else:
#                     notes.append([_beat_duration_sec, 3, note.string, 0, 0, 0])
#             else:
#                 if note.effect.slides:
#                     if note.effect.slides[0].value != 2:
#                         raise NotImplementedError
#                     notes.append([_beat_duration_sec, 1, note.string, note.value, 1, 0])
#                     continue
#                 if note.effect.grace:
#                     if not note.effect.grace.isOnBeat:
#                         notes[-1][0] = notes[-1][0] - (4 * 60 / bpm) * (1 / note.effect.grace.duration)
#                     notes.append([(4 * 60 / bpm) * (1 / note.effect.grace.duration) * 1, 1, note.string, note.effect.grace.fret, 1, 0])
#                     # notes.append([(4 * 60 / bpm) * (1 / note.effect.grace.duration) * 2, 1, note.string, note.effect.grace.fret, 1, 0])
#                     notes.append([_beat_duration_sec - (4 * 60 / bpm) * (1 / note.effect.grace.duration), 1, note.string, note.value, 0, 0])
#                     continue
#                 if note.effect.bend:
#                     notes.append([_beat_duration_sec, 1, note.string, note.value, 0, 1])
#                     continue
#                 notes.append([_beat_duration_sec, 1, note.string, note.value, 0, 0])


durations = np.array(durations)
# one and a half measure and a dotted sixteenth note
# notes = np.vstack(([4 * 60 / bpm * (1.5 + 1.5*1/32), 0, 0, 0], notes))

# tied_idxes = np.argwhere(notes[:, 1] == 2)[:, 0]
# for idx in np.flip(tied_idxes):
#     notes[idx - 1, 0] = notes[idx - 1, 0] + notes[idx, 0]
# notes = np.delete(notes, tied_idxes, axis=0)

np.save('durations.npy', durations)