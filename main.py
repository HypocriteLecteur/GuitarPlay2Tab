from manim import *
from guitarpro import parse, BeatStatus, NoteType, SlideType

from fretboard.note import Note
from fretboard.building_blocks import NotesRectangle, NotesSlack
from fretboard.fretboard import FretBoard

import numpy as np
from itertools import tee
from utils.color import *


def demonstrate_scale_structure(scene, fb, song, mode=None):
    if mode:
        raise NotImplementedError

    sample_rectangle = NotesRectangle().move_to(np.array([-2, -2, 0]))
    sample_slack = NotesSlack().move_to(np.array([2, -2, 0]))
    scene.play(FadeIn(sample_rectangle, shift=0.5*UP))
    scene.play(FadeIn(sample_slack, shift=0.5 * UP))
    scene.wait(1)

    text_key = Text("G Dorian", font_size=50, color=BLACK).shift(UP*3)
    scene.play(Write(text_key))
    scene.wait(1)

    text_roots = Text("Root", font_size=25, color=BLACK).shift(UP*2)
    scene.play(Write(text_roots))
    scene.wait(1)

    roots = fb.note_group_from_positions(fb.find_notes(Note.G), interval="1")
    scene.play(AnimationGroup(*[FadeIn(root, shift=0.5*UP) for root in roots], lag_ratio=0.1))
    scene.wait(1)

    scene.play(Unwrite(text_roots))
    scene.wait(1)

    text_triads = Text("Triad", font_size=25, color=BLACK).shift(UP*2)
    scene.play(Write(text_triads))
    scene.wait(1)

    thirds = fb.note_group_from_positions(fb.find_notes(Note.G+3), interval="b3")
    scene.play(AnimationGroup(*[FadeIn(third, shift=0.5 * UP) for third in thirds], lag_ratio=0.1))
    fifths = fb.note_group_from_positions(fb.find_notes(Note.G+7), interval="5")
    scene.play(AnimationGroup(*[FadeIn(fifth, shift=0.5 * UP) for fifth in fifths], lag_ratio=0.1))
    scene.wait(1)

    scene.play(Unwrite(text_triads))
    scene.wait(1)

    text_penta = Text("Pentatonic Scale", font_size=25, color=BLACK).shift(UP*2)
    scene.play(Write(text_penta))
    scene.wait(1)

    rectangles_lines = fb.rectangles_lines_from_positions(fb.find_notes(Note.G))
    slacks_lines = fb.slacks_lines_from_positions(fb.find_notes(Note.G))
    scene.play(AnimationGroup(*[FadeIn(a, shift=0.5 * UP) for a in rectangles_lines], lag_ratio=0))
    scene.play(AnimationGroup(*[FadeIn(a, shift=0.5 * UP) for a in slacks_lines], lag_ratio=0))
    scene.wait(1)

    fourths = fb.note_group_from_positions(fb.find_notes(Note.G + 5), interval="4")
    sevenths = fb.note_group_from_positions(fb.find_notes(Note.G - 2), interval="b7")
    foursevenths = VGroup(fourths, sevenths)
    scene.play(FadeIn(foursevenths, shift=0.5 * UP))
    scene.wait(1)

    scene.play(Unwrite(text_penta))
    scene.wait(1)

    text_mode = Text("Dorian Mode", font_size=25, color=BLACK).shift(UP*2)
    scene.play(Write(text_mode))
    scene.wait(1)

    _a = VGroup()
    _a.add(
        fb.create_note_circle(interval="2")\
            .move_to(sample_rectangle[0].get_center() + RIGHT*2*fb.fret_length)
    )
    _a.add(
        fb.create_note_circle(interval="6")\
            .move_to(sample_rectangle[0].get_center() + RIGHT*2*fb.fret_length + DOWN * fb.string_gap)
    )
    _a.add(
        fb.create_note_circle(interval="6")\
            .move_to(sample_slack[0].get_center() + LEFT*3*fb.fret_length)
    )
    _a.add(
        fb.create_note_circle(interval="2")\
            .move_to(sample_slack[0].get_center() + LEFT*3*fb.fret_length + UP * fb.string_gap)
    )
    _a.set_z_index(1)
    scene.play(FadeIn(_a, shift=0.5 * UP))
    scene.wait(1)

    seconds = fb.note_group_from_positions(fb.find_notes(Note.G + 2), interval="2")
    sixths = fb.note_group_from_positions(fb.find_notes(Note.G + 9), interval="6")
    secondsixths = VGroup(seconds, sixths)
    scene.play(FadeIn(secondsixths, shift=0.5 * UP))
    scene.wait(1)

    scene.play(Unwrite(text_mode))

    scene.play(FadeOut(slacks_lines))
    scene.play(FadeOut(rectangles_lines))

    notes_in_song = []
    for i, measure in enumerate(song.tracks[0].measures):
        for beat in measure.voices[0].beats:
            for note in beat.notes:
                if note.value > 0:
                    notes_in_song.append([note.string, note.value])
    notes_in_song = np.array(notes_in_song)
    unique_notes_in_songs = np.unique(notes_in_song, axis=0)
    a = VGroup()
    for position in unique_notes_in_songs:
        if fb.note_difference(position, Note.G):
            a.add(
                fb.create_note_circle(interval=fb.note_difference(position, Note.G), string_fret=position)
            )
    a.set_z_index(1)
    scene.add(a)

    scene.play(FadeOut(secondsixths), FadeOut(_a))
    scene.play(FadeOut(foursevenths))
    scene.play(FadeOut(fifths))
    scene.play(FadeOut(thirds))
    scene.play(FadeOut(sample_rectangle))
    scene.play(FadeOut(sample_slack))
    return


def adjust_duration(durations, song):
    # not-on-beat grace starts early
    for i, (duration, beat) in enumerate(zip(durations, get_beat(song))):
        if len(beat.notes) > 1:
            raise NotImplementedError
        if beat.notes and beat.notes[0].effect.grace and not beat.notes[0].effect.grace.isOnBeat:
            grace_duration_sec = beat.duration.value / beat.notes[0].effect.grace.duration * durations[i]
            durations[i-1] = durations[i-1] - grace_duration_sec
            durations[i] = durations[i] + grace_duration_sec

    # adjust duration to have integer frame animations (important!!!)
    duration_frames_res, duration_frames = np.modf(durations * config['frame_rate'])
    for i, duration_frame in enumerate(duration_frames_res[:-1]):
        if duration_frame >= 0.5:
            duration_frames_res[i + 1] = duration_frames_res[i + 1] - (1 - duration_frame)
            duration_frames_res[i] = 1
        else:
            duration_frames_res[i + 1] = duration_frames_res[i + 1] + duration_frame
            duration_frames_res[i] = 0
    if duration_frames_res[-1] > 0:
        duration_frames_res[-1] = 1
    durations = (duration_frames + duration_frames_res) / config['frame_rate']
    return durations


def get_beat(song):
    for measure in song.tracks[0].measures:
        for beat in measure.voices[0].beats:
            yield beat


def get_bend_point(points):
    for point in points:
        yield point


def get_two_iters(iter):
    now_it, next_it = tee(iter)
    next(next_it, None)
    return now_it, next_it


class Animation(Scene):
    def construct(self):
        # parameters -----------------------------------------------------
        song = parse('D:\\guitar\\animation\\ALONE.gp5')

        durations = np.load('D:\\guitar\\animation\\durations.npy')
        durations = adjust_duration(durations, song)

        background = ImageMobject("D:\\guitar\\animation\\merge.png")
        background.scale(2)
        
        # initialization--------------------------------------------------
        self.add(background)
        fb = FretBoard().shift(DOWN*0.5)
        self.add(fb)

        # song structure--------------------------------------------------
        self.next_section(skip_animations=True)
        demonstrate_scale_structure(self, fb, song)

        # tab animations--------------------------------------------------
        self.next_section()
        prev_beat = None
        now_beat_iter, next_beat_iter = get_two_iters(get_beat(song))
        for duration_sec, now_beat, future_beat in zip(durations, now_beat_iter, next_beat_iter):
            if now_beat.status == BeatStatus.empty or now_beat.status == BeatStatus.rest:
                self.wait(duration_sec, frozen_frame=True)
                continue
            
            # assume one note per beat
            if len(now_beat.notes) > 1:
                raise NotImplementedError

            if prev_beat is not None and now_beat.notes[0].type == NoteType.tie:  # tied beats share the same fret
                    now_beat.notes[0].value = prev_beat.notes[0].value

            if now_beat.notes[0].effect.slides:
                if now_beat.notes[0].effect.slides[0] != SlideType.legatoSlideTo:
                    raise NotImplementedError
                _note = fb.create_playing_note_circle((now_beat.notes[0].string, now_beat.notes[0].value))
                self.add(_note)
                self.play(_note.animate
                            .shift(LEFT * (now_beat.notes[0].value - future_beat.notes[0].value) * fb.fret_length),
                            run_time=duration_sec,
                            rate_functions=rate_functions.linear)
            elif now_beat.notes[0].effect.grace:
                _note = fb.create_playing_note_circle((now_beat.notes[0].string, now_beat.notes[0].effect.grace.fret))
                self.add(_note)

                grace_duration_percentage = now_beat.duration.value / now_beat.notes[0].effect.grace.duration
                grace_duration_frame = round(duration_sec * grace_duration_percentage * config['frame_rate'])
                self.wait(grace_duration_frame / config['frame_rate'])

                self.play(_note.animate
                            .shift(LEFT * (now_beat.notes[0].effect.grace.fret - now_beat.notes[0].value) * fb.fret_length),
                            run_time=(duration_sec * config['frame_rate'] - grace_duration_frame) /
                                    config['frame_rate'], rate_functions=rate_functions.linear)

            elif now_beat.notes[0].effect.bend:
                string_id = 6 - now_beat.notes[0].string

                fb.strings[string_id].set_opacity(0)
                fb.strings[string_id-1].set_opacity(0)
                _note = fb.create_playing_note_circle((now_beat.notes[0].string, now_beat.notes[0].value))

                line = always_redraw(
                    lambda: VGroup(
                        Line(fb.string_fret_to_pos(now_beat.notes[0].string, 0) + RIGHT * 0.5 * fb.fret_length, _note.get_center(), color=BLACK),
                        Line(_note.get_center(), fb.string_fret_to_pos(now_beat.notes[0].string, fb.frets_number) + RIGHT * 0.5 * fb.fret_length, color=BLACK)
                    )
                )
                line2 = always_redraw(
                    lambda: VGroup(
                        Line(fb.string_fret_to_pos(now_beat.notes[0].string+1, 0) + RIGHT * 0.5 * fb.fret_length, np.array([_note.get_center()[0], np.min((_note.get_center()[1], fb.strings[string_id-1].get_center()[1])), 0]), color=BLACK),
                        Line(np.array([_note.get_center()[0], np.min((_note.get_center()[1], fb.strings[string_id-1].get_center()[1])), 0]), fb.string_fret_to_pos(now_beat.notes[0].string+1, fb.frets_number) + RIGHT * 0.5 * fb.fret_length, color=BLACK)
                    )
                )
                self.add(_note, line, line2)

                _note.shift(DOWN * now_beat.notes[0].effect.bend.points[0].value/2 * fb.string_gap)

                _it1, _it2 = tee(get_bend_point(now_beat.notes[0].effect.bend.points))
                next(_it2, None)
                _durations = []
                for now_point, next_point in zip(_it1, _it2):
                    if next_point is not None:
                        _durations.append((next_point.position - now_point.position)/12*duration_sec)
                for i, _duration in enumerate(_durations[:-1]):
                    _durations[i] = round(_duration * config['frame_rate']) / config['frame_rate']
                _durations = np.array(_durations)
                _durations[-1] = duration_sec - np.sum(_durations[:-1])

                _it1, _it2 = tee(get_bend_point(now_beat.notes[0].effect.bend.points))
                next(_it2, None)
                for i, (now_point, next_point) in enumerate(zip(_it1, _it2)):
                    if next_point is not None:
                        self.play(_note.animate.shift(DOWN * (next_point.value - now_point.value)/2 * fb.string_gap),
                                    run_time=_durations[i], rate_functions=rate_functions.linear)

                self.remove(line, line2)
                fb.strings[string_id].set_opacity(1)
                fb.strings[string_id - 1].set_opacity(1)
            else:
                _note = fb.create_playing_note_circle((now_beat.notes[0].string, now_beat.notes[0].value))
                self.add(_note)
                self.wait(duration_sec, frozen_frame=True)
            self.remove(_note)
            prev_beat = now_beat


with tempconfig({"quality": "low_quality", "disable_caching": True, 'frame_rate': 30, 'preview': True}):
    scene = Animation()
    scene.render()