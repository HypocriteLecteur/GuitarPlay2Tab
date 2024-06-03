from manim import *
from guitarpro import parse

from fretboard.note import Note
from fretboard.building_blocks import NotesRectangle, NotesSlack
from fretboard.fretboard import FretBoard

import numpy as np
from itertools import tee
from utils.color import *


def adjust_duration(durations, song, bpm):
    # grace
    for i, (duration, beat) in enumerate(zip(durations, get_beat(song))):
        if len(beat.notes) > 1:
            raise NotImplementedError
        if beat.notes and beat.notes[0].effect.grace and not beat.notes[0].effect.grace.isOnBeat:
            durations[i-1] = durations[i-1] - beat.duration.value / beat.notes[0].effect.grace.duration * durations[i]
            durations[i] = durations[i] + beat.duration.value / beat.notes[0].effect.grace.duration * durations[i]

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


class Animation(Scene):
    def construct(self):
        # Create a number line which will be used as a progress bar
        n = NumberLine(include_numbers=True, x_range=[0,12]).to_edge(DOWN)
        self.add(n)

        background = ImageMobject("D:\\guitar\\animation\\merge.png")
        background.scale(2)
        self.add(background)

        fb = FretBoard().shift(DOWN*0.5)
        self.add(fb)

        self.next_section(skip_animations=True)
        # ----------------------------------------------------------------------
        song = parse('D:\\guitar\\animation\\ALONE.gp5')
        bpm = 116.6  # part 1

        durations = np.load('D:\\guitar\\animation\\durations.npy')
        durations = adjust_duration(durations, song, bpm)

        notes_in_song = []
        for i, measure in enumerate(song.tracks[0].measures):
            for beat in measure.voices[0].beats:
                # check for not implemented features
                for note in beat.notes:
                    if note.value > 0:
                        notes_in_song.append([note.string, note.value])
        notes_in_song = np.array(notes_in_song)

        sample_rectangle = NotesRectangle().move_to(np.array([-2, -2, 0]))
        sample_slack = NotesSlack().move_to(np.array([2, -2, 0]))
        self.play(FadeIn(sample_rectangle, shift=0.5*UP))
        self.play(FadeIn(sample_slack, shift=0.5 * UP))

        self.wait(1)

        text_key = Text("G Dorian", font_size=50, color=BLACK).shift(UP*3)
        self.play(Write(text_key))

        self.wait(1)

        text_roots = Text("Root", font_size=25, color=BLACK).shift(UP*2)
        self.play(Write(text_roots))
        self.wait(1)

        roots = fb.note_group_from_positions(fb.find_notes(Note.G), interval="1")
        self.play(AnimationGroup(*[FadeIn(root, shift=0.5*UP) for root in roots], lag_ratio=0.1))

        self.wait(1)

        self.play(Unwrite(text_roots))
        self.wait(1)
        text_triads = Text("Triad", font_size=25, color=BLACK).shift(UP*2)
        self.play(Write(text_triads))
        self.wait(1)

        thirds = fb.note_group_from_positions(fb.find_notes(Note.G+3), interval="b3")
        self.play(AnimationGroup(*[FadeIn(third, shift=0.5 * UP) for third in thirds], lag_ratio=0.1))

        fifths = fb.note_group_from_positions(fb.find_notes(Note.G+7), interval="5")
        self.play(AnimationGroup(*[FadeIn(fifth, shift=0.5 * UP) for fifth in fifths], lag_ratio=0.1))

        self.wait(1)

        self.play(Unwrite(text_triads))
        self.wait(1)
        text_penta = Text("Pentatonic Scale", font_size=25, color=BLACK).shift(UP*2)
        self.play(Write(text_penta))
        self.wait(1)

        rectangles_lines = fb.rectangles_lines_from_positions(fb.find_notes(Note.G))
        slacks_lines = fb.slacks_lines_from_positions(fb.find_notes(Note.G))
        self.play(AnimationGroup(*[FadeIn(a, shift=0.5 * UP) for a in rectangles_lines], lag_ratio=0))
        self.play(AnimationGroup(*[FadeIn(a, shift=0.5 * UP) for a in slacks_lines], lag_ratio=0))

        self.wait(1)

        # self.play(Unwrite(text_penta))
        # self.wait(1)
        # text_mode = Text("Mode", font_size=25, color=BLACK).shift(UP*2)
        # self.play(Write(text_mode))
        # self.wait(1)

        fourths = fb.note_group_from_positions(fb.find_notes(Note.G + 5), interval="4")
        sevenths = fb.note_group_from_positions(fb.find_notes(Note.G - 2), interval="b7")
        foursevenths = VGroup(fourths, sevenths)

        seconds = fb.note_group_from_positions(fb.find_notes(Note.G + 2), interval="2")
        sixths = fb.note_group_from_positions(fb.find_notes(Note.G + 9), interval="6")
        secondsixths = VGroup(seconds, sixths)

        self.play(FadeIn(foursevenths, shift=0.5 * UP))

        self.wait(1)

        self.play(Unwrite(text_penta))
        self.wait(1)
        text_mode = Text("Dorian Mode", font_size=25, color=BLACK).shift(UP*2)
        self.play(Write(text_mode))
        self.wait(1)

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
        self.play(FadeIn(_a, shift=0.5 * UP))

        self.wait(1)
        self.play(FadeIn(secondsixths, shift=0.5 * UP))

        self.wait(1)

        self.play(Unwrite(text_mode))

        self.play(FadeOut(slacks_lines))
        self.play(FadeOut(rectangles_lines))

        unique_notes_in_songs = np.unique(notes_in_song, axis=0)
        a = VGroup()
        for position in unique_notes_in_songs:
            if fb.note_difference(position, Note.G):
                a.add(
                    fb.create_note_circle(interval=fb.note_difference(position, Note.G), string_fret=position)
                )
        a.set_z_index(1)
        self.add(a)

        self.play(FadeOut(secondsixths), FadeOut(_a))
        self.play(FadeOut(foursevenths))
        self.play(FadeOut(fifths))
        self.play(FadeOut(thirds))
        self.play(FadeOut(sample_rectangle))
        self.play(FadeOut(sample_slack))

        self.next_section()

        # -------------------------------------------------------------------------------------------
        # progress_bar = Line(n.n2p(0), n.n2p(12)).set_stroke(color=YELLOW, opacity=0.5, width=20)
        # Create a timeline of animations. One of the animations is the progress bar itself
        # going from 0 to 12 in 12 seconds. It can be used as a reference to check that
        # the other animations are playing at the right time.
        # timeline = {
        #     0: Create(progress_bar, run_time=12, rate_func=linear),
        #     1: Create(Square(), run_time=10),
        #     2: [Create(Circle(), run_time=4),
        #         Create(Triangle(), run_time=2)],
        #     9: Write(Text("It works!").next_to(progress_bar, UP, buff=1),
        #              run_time=3)
        # }
        # play_timeline(self, timeline)
        # self.wait()
        # -------------------------------------------------------------------------------------------

        prev_beat = None
        iter = get_beat(song)
        it1, it2 = tee(iter)
        next(it2, None)
        for duration_sec, now_beat, future_beat in zip(durations, it1, it2):
            if now_beat.status.value == 0 or now_beat.status.value == 2:
                self.wait(duration_sec, frozen_frame=True)
            else:
                # assume one note per beat
                if len(now_beat.notes) > 1:
                    raise NotImplementedError

                if prev_beat is not None and now_beat.notes[0].type.value == 2:  # tied beats share the same fret
                        now_beat.notes[0].value = prev_beat.notes[0].value

                if now_beat.notes[0].effect.slides:
                    if now_beat.notes[0].effect.slides[0].value != 2:
                        raise NotImplementedError
                    _note = fb.create_playing_note_circle((now_beat.notes[0].string, now_beat.notes[0].value))
                    self.add(_note)
                    self.play(_note.animate
                              .shift(LEFT * (now_beat.notes[0].value - future_beat.notes[0].value) * fb.fretboard_length*2/fb.frets_number),
                              run_time=duration_sec,
                              rate_functions=rate_functions.linear)
                elif now_beat.notes[0].effect.grace:
                    _note = fb.create_playing_note_circle((now_beat.notes[0].string, now_beat.notes[0].effect.grace.fret))
                    self.add(_note)

                    grace_duration_percentage = now_beat.duration.value / now_beat.notes[0].effect.grace.duration
                    grace_duration_frame = round(duration_sec * grace_duration_percentage * config['frame_rate'])
                    self.wait(grace_duration_frame / config['frame_rate'])

                    self.play(_note.animate
                              .shift(LEFT * (now_beat.notes[0].effect.grace.fret - now_beat.notes[0].value) * fb.fretboard_length * 2 / fb.frets_number),
                              run_time=(duration_sec * config['frame_rate'] - grace_duration_frame) /
                                       config['frame_rate'], rate_functions=rate_functions.linear)

                elif now_beat.notes[0].effect.bend:
                    string_id = 6 - now_beat.notes[0].string

                    fb.strings[string_id].set_opacity(0)
                    fb.strings[string_id-1].set_opacity(0)
                    _note = fb.create_playing_note_circle((now_beat.notes[0].string, now_beat.notes[0].value))

                    line = always_redraw(
                        lambda: VGroup(
                            Line(fb.string_fret_to_pos(now_beat.notes[0].string, 0) + RIGHT * 0.5 * fb.fretboard_length * 2 / fb.frets_number, _note.get_center(), color=BLACK),
                            Line(_note.get_center(), fb.string_fret_to_pos(now_beat.notes[0].string, fb.frets_number) + RIGHT * 0.5 * fb.fretboard_length * 2 / fb.frets_number, color=BLACK)
                        )
                    )
                    line2 = always_redraw(
                        lambda: VGroup(
                            Line(fb.string_fret_to_pos(now_beat.notes[0].string+1, 0) + RIGHT * 0.5 * fb.fretboard_length * 2 / fb.frets_number, np.array([_note.get_center()[0], np.min((_note.get_center()[1], fb.strings[string_id-1].get_center()[1])), 0]), color=BLACK),
                            Line(np.array([_note.get_center()[0], np.min((_note.get_center()[1], fb.strings[string_id-1].get_center()[1])), 0]), fb.string_fret_to_pos(now_beat.notes[0].string+1, fb.frets_number) + RIGHT * 0.5 * fb.fretboard_length * 2 / fb.frets_number, color=BLACK)
                        )
                    )
                    self.add(_note, line, line2)

                    # a.add_updater(
                    #     lambda x: [notecircle.move_to(np.array([notecircle.get_center()[0], , 0])) for notecircle in x]
                    # )
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