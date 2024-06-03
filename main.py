from manim import *
import numpy as np
from functools import partial
from enum import Enum
from guitarpro import parse
from itertools import tee

COLOR_1 = ManimColor("#CC5402")
COLOR_3 = ManimColor("#3C8B6C")
COLOR_5 = ManimColor("#FFC000")
COLOR_47 = ManimColor("#2465A5")

SingleText = partial(Text, font_size=15, stroke_width=1.5, fill_opacity=1, z_index=2)
DoubleText = partial(Text, font_size=10, stroke_width=1.5, fill_opacity=1, z_index=2)

NoteCircle = partial(Circle, fill_opacity=1, z_index=3)

RectangleLine = partial(Line, stroke_width=7, color=ManimColor("#FF0000"), stroke_opacity=0.6)
SlackLine = partial(Line, stroke_width=7, color=ManimColor("#0400F9"), stroke_opacity=0.6)


class Note(Enum):
    E = 0
    F = 1
    Fsharp = 2
    G = 3
    Gsharp = 4
    A = 5
    Asharp = 6
    B = 7
    C = 8
    Csharp = 9
    D = 10
    Dsharp = 11

    def __add__(self, interval):
        return Note(np.mod(self.value + interval, 12))

    def __sub__(self, interval):
        return Note(np.mod(self.value - interval, 12))


class NotesRectangle(VGroup):
    def __init__(self,
                 fretboard_length: float = 6,
                 fretboard_width: float = 0.3,
                 **kwargs):
        self.fretboard_length = fretboard_length
        self.fretboard_width = fretboard_width
        VGroup.__init__(self, **kwargs)

        note_1 = VGroup(NoteCircle(radius=0.3*fretboard_width, color=COLOR_1), SingleText("1")).set_z_index(1)
        note_b3 = VGroup(NoteCircle(radius=0.3*fretboard_width, color=COLOR_3), DoubleText("b3"), z_index=1)\
            .move_to(note_1.get_center() + np.array((3*fretboard_length*2/24, 0, 0))).set_z_index(1)
        note_5 = VGroup(NoteCircle(radius=0.3*fretboard_width, color=COLOR_5), SingleText("5"), z_index=1)\
            .move_to(note_1.get_center() + np.array((0, -fretboard_width, 0))).set_z_index(1)
        note_b7 = VGroup(NoteCircle(radius=0.3*fretboard_width, color=COLOR_47), DoubleText("b7"), z_index=1)\
            .move_to(note_5.get_center() + np.array((3*fretboard_length*2/24, 0, 0))).set_z_index(1)

        self.lines = VGroup()
        self.lines.add(RectangleLine(note_1.get_center(), note_b3.get_center()))
        self.lines.add(RectangleLine(note_1.get_center(), note_5.get_center()))
        self.lines.add(RectangleLine(note_b3.get_center(), note_b7.get_center()))
        self.lines.add(RectangleLine(note_5.get_center(), note_b7.get_center()))

        self.add(note_1)
        self.add(note_b3)
        self.add(note_5)
        self.add(note_b7)
        self.add(self.lines)


class NotesSlack(VGroup):
    def __init__(self,
                 fretboard_length: float = 6,
                 fretboard_width: float = 0.3,
                 **kwargs):
        self.fretboard_length = fretboard_length
        self.fretboard_width = fretboard_width
        VGroup.__init__(self, **kwargs)

        note_1 = VGroup(NoteCircle(radius=0.3*fretboard_width, color=COLOR_1), SingleText("1")).set_z_index(1)
        note_b3 = VGroup(NoteCircle(radius=0.3*fretboard_width, color=COLOR_3), DoubleText("b3")).set_z_index(1)\
            .move_to(note_1.get_center() + np.array((-2 * fretboard_length * 2 / 24, fretboard_width, 0)))
        note_5 = VGroup(NoteCircle(radius=0.3*fretboard_width, color=COLOR_5), SingleText("5")).set_z_index(1)\
            .move_to(note_1.get_center() + np.array((0, -fretboard_width, 0)))
        note_4l = VGroup(NoteCircle(radius=0.3*fretboard_width, color=COLOR_47), SingleText("4")).set_z_index(1)\
            .move_to(note_5.get_center() + np.array((-2*fretboard_length*2/24, 0, 0)))
        note_4h = VGroup(NoteCircle(radius=0.3*fretboard_width, color=COLOR_47), SingleText("4")).set_z_index(1)\
            .move_to(note_1.get_center() + np.array((0, fretboard_width, 0)))
        note_b7 = VGroup(NoteCircle(radius=0.3*fretboard_width, color=COLOR_47), DoubleText("b7")).set_z_index(1)\
            .move_to(note_1.get_center() + np.array((-2*fretboard_length*2/24, 0, 0)))

        self.lines = VGroup()
        self.lines.add(SlackLine(note_1.get_center(), note_5.get_center()))
        self.lines.add(SlackLine(note_1.get_center(), note_4h.get_center()))
        self.lines.add(SlackLine(note_4l.get_center(), note_5.get_center()))
        self.lines.add(SlackLine(note_b7.get_center(), note_4l.get_center()))
        self.lines.add(SlackLine(note_b7.get_center(), note_b3.get_center()))
        self.lines.add(SlackLine(note_b3.get_center(), note_4h.get_center()))

        self.add(note_1)
        self.add(note_b3)
        self.add(note_5)
        self.add(note_4l)
        self.add(note_4h)
        self.add(note_b7)
        self.add(self.lines)


class FretBoard(VGroup):
    def __init__(self,
                 strings_number: int = 6,
                 frets_number: int = 24,
                 tuning: str = 'E standard',
                 fretboard_length: float = 6,
                 fretboard_width: float = 0.3,
                 fretboard_color: ManimColor = BLACK,
                 **kwargs):
        self.strings_number = strings_number
        self.frets_number = frets_number
        self.fretboard_length = fretboard_length
        self.fretboard_width = fretboard_width
        self.fretboard_color = fretboard_color
        self.tuning = tuning
        VGroup.__init__(self, **kwargs)

        # -------------------------------------------------
        self.strings = VGroup()
        for i in range(self.strings_number):
            self.strings.add(Line(np.array((-fretboard_length, i*fretboard_width, 0.0)),
                                  np.array((fretboard_length, i*fretboard_width, 0.0)),
                                  stroke_width=2.5 - i*0.15, color=self.fretboard_color))
        self.add(self.strings)

        # -------------------------------------------------
        self.fret_lines = VGroup()
        for i in range(self.frets_number + 1):
            self.fret_lines.add(Line(np.array((-fretboard_length + i*fretboard_length*2/self.frets_number, 0, 0.0)),
                                     np.array((-fretboard_length + i*fretboard_length*2/self.frets_number, (self.strings_number - 1)*fretboard_width, 0.0)),
                                     stroke_width=2.5, color=self.fretboard_color))
        self.add(self.fret_lines)

        # -------------------------------------------------
        self.dots = VGroup()

        offset = np.array((0, 0.5*fretboard_width, 0))
        dots_pos = [(4, 3), (4, 5), (4, 7), (4, 9), (5, 12), (3, 12),
                    (4, 15), (4, 17), (4, 19), (4, 21), (5, 24), (3, 24)]
        for dot_pos in dots_pos:
            self.dots.add(Circle(radius=0.2 * fretboard_width, color=GREY, fill_opacity=1)
                          .move_to(self.string_fret_coords_to_pos(*dot_pos) + offset))

        self.add(self.dots)

        # -------------------------------------------------
        if tuning == 'E standard':
            self.strings_offsets = np.array([5, 5, 5, 4, 5])
        else:
            raise NotImplementedError

    def string_fret_coords_to_pos(self, string_num, fret_num):
        string_id = self.strings_number - string_num

        bottom_string_y = self.strings[0].get_center()[1]
        string_gap = self.strings[1].get_center()[1] - bottom_string_y
        y = bottom_string_y + string_gap * string_id


        fret_id = fret_num
        zero_fret_x = self.fret_lines[0].get_center()[0]
        fret_gap = self.fret_lines[1].get_center()[0] - zero_fret_x
        offset = (self.fret_lines[1].get_center()[0] - self.fret_lines[0].get_center()[0]) / 2
        x = zero_fret_x + fret_gap*fret_id - offset
        return np.array((x, y, 0))

    def note_difference(self, location, note):
        string_difference = int(abs(self.strings_number - location[0]))
        return int(np.mod(np.sum(self.strings_offsets[:string_difference]) + location[1] - note.value, 12))

    def confine_to_fretboard(self, group_position):
        group_position = group_position[group_position[:, 0] >= 1, :]
        group_position = group_position[group_position[:, 0] <= self.strings_number, :]
        group_position = group_position[group_position[:, 1] <= self.frets_number, :]
        group_position = group_position[group_position[:, 1] >= 0, :]
        return group_position

    def find_notes(self, note: Note):
        if self.tuning != 'E standard':
            raise NotImplementedError

        base_position = np.array((self.strings_number, note.value))

        positions = np.empty((0, 2)).astype(int)
        for string in range(self.strings_number, 0, -1):
            for fret in range(self.frets_number+1):
                string_difference = abs(string - base_position[0])
                fret_difference = fret - base_position[1]
                interval_difference = np.sum(self.strings_offsets[:string_difference]) + fret_difference
                if np.mod(interval_difference, 12) == 0:
                    positions = np.vstack((positions, [string, fret]))
        return positions

    def build_minor_triads(self, root: Note):
        root_positions = self.find_notes(root)
        minor_third_positions = self.find_notes(root+3)
        perfect_fifth_positions = self.find_notes(root+7)
        return root_positions, minor_third_positions, perfect_fifth_positions

    def rectangles_lines_from_positions(self, positions):
        if self.tuning != 'E standard':
            raise NotImplementedError

        rectangles = VGroup()
        for position in positions:
            if position[0] == self.strings_number:
                rectangles.add(RectangleLine(
                    self.string_fret_coords_to_pos(position[0], position[1]),
                    self.string_fret_coords_to_pos(position[0], position[1]+3)
                ))
                continue
            elif position[0] == 2:
                rectangle_positions = np.array([
                    [0, 0],
                    [0, 3],
                    [1, 2],
                    [1, -1]
                ])
            else:
                rectangle_positions = np.array([
                    [0, 0],
                    [0, 3],
                    [1, 3],
                    [1, 0]
                ])
            rectangle_positions = rectangle_positions + position

            idxes = (rectangle_positions[:, 0] >= 1) & (rectangle_positions[:, 0] <= self.strings_number) & \
                    (rectangle_positions[:, 1] >= 0) & (rectangle_positions[:, 1] <= self.frets_number)
            idxes = idxes & np.roll(idxes, -1)
            pairs = np.array(((0, 1), (1, 2), (2, 3), (3, 0)))

            rectangle = VGroup()
            for pair in pairs[idxes]:
                rectangle.add(RectangleLine(self.string_fret_coords_to_pos(rectangle_positions[pair[0], 0], rectangle_positions[pair[0], 1]),
                                            self.string_fret_coords_to_pos(rectangle_positions[pair[1], 0], rectangle_positions[pair[1], 1])))
            rectangles.add(rectangle)

        # add in the missing lines

        return rectangles

    def slacks_lines_from_positions(self, positions):
        if self.tuning != 'E standard':
            raise NotImplementedError

        slacks = VGroup()
        for position in positions:
            if position[0] == 3:
                slack_positions = np.array([
                    [0, 0],
                    [1, 0],
                    [1, -2],
                    [0, -2],
                    [-1, -1],
                    [-1, 1]
                ])
            elif position[0] == 2:
                slack_positions = np.array([
                    [0, 0],
                    [1, -1],
                    [1, -3],
                    [0, -2],
                    [-1, -2],
                    [-1, 0]
                ])
            else:
                slack_positions = np.array([
                    [0, 0],
                    [1, 0],
                    [1, -2],
                    [0, -2],
                    [-1, -2],
                    [-1, 0]
                ])
            slack_positions = slack_positions + position

            idxes = (slack_positions[:, 0] >= 1) & (slack_positions[:, 0] <= self.strings_number) & \
                    (slack_positions[:, 1] >= 0) & (slack_positions[:, 1] <= self.frets_number)
            idxes = idxes & np.roll(idxes, -1)
            pairs = np.array(((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)))

            slack = VGroup()
            for pair in pairs[idxes]:
                slack.add(SlackLine(self.string_fret_coords_to_pos(slack_positions[pair[0], 0], slack_positions[pair[0], 1]),
                                    self.string_fret_coords_to_pos(slack_positions[pair[1], 0], slack_positions[pair[1], 1])))
            slacks.add(slack)
        return slacks

    def note_group_from_positions(self, positions, color, text):
        note_group = VGroup()
        if len(text) == 1:
            for position in positions:
                note_group.add(VGroup(NoteCircle(radius=0.3*self.fretboard_width, color=color),
                                      SingleText(text)).move_to(self.string_fret_coords_to_pos(position[0], position[1])))
        else:
            for position in positions:
                note_group.add(VGroup(NoteCircle(radius=0.3 * self.fretboard_width, color=color),
                                      DoubleText(text)).move_to(
                    self.string_fret_coords_to_pos(position[0], position[1])))
        return note_group.set_z_index(1)


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

        background = ImageMobject("merge.png")
        background.scale(2)
        self.add(background)

        fb = FretBoard().shift(DOWN*0.5)
        self.add(fb)

        self.next_section(skip_animations=True)
        # ----------------------------------------------------------------------
        song = parse('ALONE.gp5')
        bpm = 116.6  # part 1

        durations = np.load('durations.npy')
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

        roots = fb.note_group_from_positions(fb.find_notes(Note.G), color=COLOR_1, text='1')
        self.play(AnimationGroup(*[FadeIn(root, shift=0.5*UP) for root in roots], lag_ratio=0.1))

        self.wait(1)

        self.play(Unwrite(text_roots))
        self.wait(1)
        text_triads = Text("Triad", font_size=25, color=BLACK).shift(UP*2)
        self.play(Write(text_triads))
        self.wait(1)

        thirds = fb.note_group_from_positions(fb.find_notes(Note.G+3), color=COLOR_3, text='b3')
        self.play(AnimationGroup(*[FadeIn(third, shift=0.5 * UP) for third in thirds], lag_ratio=0.1))

        fifths = fb.note_group_from_positions(fb.find_notes(Note.G+7), color=COLOR_5, text='5')
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

        fourths = fb.note_group_from_positions(fb.find_notes(Note.G + 5), color=COLOR_47, text='4')
        sevenths = fb.note_group_from_positions(fb.find_notes(Note.G - 2), color=COLOR_47, text='b7')
        foursevenths = VGroup(fourths, sevenths)

        seconds = fb.note_group_from_positions(fb.find_notes(Note.G + 2), color=BLACK, text='2')
        sixths = fb.note_group_from_positions(fb.find_notes(Note.G + 9), color=BLACK, text='6')
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
            VGroup(
                NoteCircle(radius=0.3 * fb.fretboard_width, color=BLACK),
                SingleText('2')
            ).move_to(sample_rectangle[0].get_center() + RIGHT*2*fb.fretboard_length*2/fb.frets_number)
        )
        _a.add(
            VGroup(
                NoteCircle(radius=0.3 * fb.fretboard_width, color=BLACK),
                SingleText('6')
            ).move_to(sample_rectangle[0].get_center() + RIGHT*2*fb.fretboard_length*2/fb.frets_number + DOWN * fb.fretboard_width)
        )
        _a.add(
            VGroup(
                NoteCircle(radius=0.3 * fb.fretboard_width, color=BLACK),
                SingleText('6')
            ).move_to(sample_slack[0].get_center() + LEFT*3*fb.fretboard_length*2/fb.frets_number)
        )
        _a.add(
            VGroup(
                NoteCircle(radius=0.3 * fb.fretboard_width, color=BLACK),
                SingleText('2')
            ).move_to(sample_slack[0].get_center() + LEFT*3*fb.fretboard_length*2/fb.frets_number + UP * fb.fretboard_width)
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
            diff = fb.note_difference(position, Note.G)
            if diff == 3:
                a.add(
                    VGroup(
                        NoteCircle(radius=0.3 * fb.fretboard_width, color=COLOR_3),
                        DoubleText('b3')
                    ).move_to(fb.string_fret_coords_to_pos(position[0], position[1]))
                )
            elif diff == 7:
                a.add(
                    VGroup(
                        NoteCircle(radius=0.3 * fb.fretboard_width, color=COLOR_5),
                        SingleText('5')
                    ).move_to(fb.string_fret_coords_to_pos(position[0], position[1]))
                )
            elif diff == 5:
                a.add(
                    VGroup(
                        NoteCircle(radius=0.3 * fb.fretboard_width, color=COLOR_47),
                        SingleText('4')
                    ).move_to(fb.string_fret_coords_to_pos(position[0], position[1]))
                )
            elif diff == 10:
                a.add(
                    VGroup(
                        NoteCircle(radius=0.3 * fb.fretboard_width, color=COLOR_47),
                        DoubleText('b7')
                    ).move_to(fb.string_fret_coords_to_pos(position[0], position[1]))
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
                    _note = NoteCircle(radius=0.5 * fb.fretboard_width, color=RED, stroke_color=BLACK) \
                        .move_to(
                        fb.string_fret_coords_to_pos(now_beat.notes[0].string, now_beat.notes[0].value)).set_z_index(3)
                    _note.set_opacity(0.6)
                    self.add(_note)
                    self.play(_note.animate
                              .shift(LEFT * (now_beat.notes[0].value - future_beat.notes[0].value) * fb.fretboard_length*2/fb.frets_number),
                              run_time=duration_sec,
                              rate_functions=rate_functions.linear)
                elif now_beat.notes[0].effect.grace:
                    _note = NoteCircle(radius=0.5 * fb.fretboard_width, color=RED, stroke_color=BLACK) \
                        .move_to(
                        fb.string_fret_coords_to_pos(now_beat.notes[0].string, now_beat.notes[0].effect.grace.fret)).set_z_index(3)
                    _note.set_opacity(0.6)
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
                    _note = NoteCircle(radius=0.5 * fb.fretboard_width, color=RED, stroke_color=BLACK) \
                        .move_to(
                        fb.string_fret_coords_to_pos(now_beat.notes[0].string, now_beat.notes[0].value)).set_z_index(3)
                    _note.set_opacity(0.6)

                    line = always_redraw(
                        lambda: VGroup(
                            Line(fb.string_fret_coords_to_pos(now_beat.notes[0].string, 0) + RIGHT * 0.5 * fb.fretboard_length * 2 / fb.frets_number, _note.get_center(), color=BLACK),
                            Line(_note.get_center(), fb.string_fret_coords_to_pos(now_beat.notes[0].string, fb.frets_number) + RIGHT * 0.5 * fb.fretboard_length * 2 / fb.frets_number, color=BLACK)
                        )
                    )
                    line2 = always_redraw(
                        lambda: VGroup(
                            Line(fb.string_fret_coords_to_pos(now_beat.notes[0].string+1, 0) + RIGHT * 0.5 * fb.fretboard_length * 2 / fb.frets_number, np.array([_note.get_center()[0], np.min((_note.get_center()[1], fb.strings[string_id-1].get_center()[1])), 0]), color=BLACK),
                            Line(np.array([_note.get_center()[0], np.min((_note.get_center()[1], fb.strings[string_id-1].get_center()[1])), 0]), fb.string_fret_coords_to_pos(now_beat.notes[0].string+1, fb.frets_number) + RIGHT * 0.5 * fb.fretboard_length * 2 / fb.frets_number, color=BLACK)
                        )
                    )
                    self.add(_note, line, line2)

                    # a.add_updater(
                    #     lambda x: [notecircle.move_to(np.array([notecircle.get_center()[0], , 0])) for notecircle in x]
                    # )
                    _note.shift(DOWN * now_beat.notes[0].effect.bend.points[0].value/2 * fb.fretboard_width)

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
                            # self.wait(_durations[i], frozen_frame=False)
                            self.play(_note.animate.shift(DOWN * (next_point.value - now_point.value)/2 * fb.fretboard_width),
                                      run_time=_durations[i], rate_functions=rate_functions.linear)

                    self.remove(line, line2)
                    fb.strings[string_id].set_opacity(1)
                    fb.strings[string_id - 1].set_opacity(1)
                else:
                    _note = NoteCircle(radius=0.5 * fb.fretboard_width, color=RED, stroke_color=BLACK) \
                        .move_to(
                        fb.string_fret_coords_to_pos(now_beat.notes[0].string, now_beat.notes[0].value)).set_z_index(3)
                    _note.set_opacity(0.6)
                    self.add(_note)
                    self.wait(duration_sec, frozen_frame=True)
                self.remove(_note)
                prev_beat = now_beat


with tempconfig({"quality": "low_quality", "disable_caching": True, 'frame_rate': 30}):
    scene = Animation()
    scene.render()