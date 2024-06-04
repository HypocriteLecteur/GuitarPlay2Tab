from guitarpro import parse, Song

import numpy as np

def get_bend_points_iter(points):
    for point in points:
        yield point

def get_beats_iter(song: Song):
    for measure in song.tracks[0].measures:
        for beat in measure.voices[0].beats:
            yield beat

def extract_durations(song: Song, bpm=None):
    if bpm is None:
        bpm = song.tempo
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
    # np.save('durations.npy', durations)
    return durations

def adjust_duration(durations, song: Song, frame_rate:int):
    # not-on-beat grace starts early
    for i, (duration, beat) in enumerate(zip(durations, get_beats_iter(song))):
        if len(beat.notes) > 1:
            raise NotImplementedError
        if beat.notes and beat.notes[0].effect.grace and not beat.notes[0].effect.grace.isOnBeat:
            grace_duration_sec = beat.duration.value / beat.notes[0].effect.grace.duration * durations[i]
            durations[i-1] = durations[i-1] - grace_duration_sec
            durations[i] = durations[i] + grace_duration_sec
    
    # adjust duration to have integer frame animations (important!!!)
    duration_frames_res, duration_frames = np.modf(durations * frame_rate)
    for i, duration_frame in enumerate(duration_frames_res[:-1]):
        if duration_frame >= 0.5:
            duration_frames_res[i + 1] = duration_frames_res[i + 1] - (1 - duration_frame)
            duration_frames_res[i] = 1
        else:
            duration_frames_res[i + 1] = duration_frames_res[i + 1] + duration_frame
            duration_frames_res[i] = 0
    if duration_frames_res[-1] > 0:
        duration_frames_res[-1] = 1
    durations = (duration_frames + duration_frames_res) / frame_rate
    return durations