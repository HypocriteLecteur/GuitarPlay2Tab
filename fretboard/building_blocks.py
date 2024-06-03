from manim import Text, Circle, Line, VGroup
from utils.color import *
from functools import partial
import numpy as np

SingleText = partial(Text, font_size=15, stroke_width=1.5, fill_opacity=1, z_index=1)
DoubleText = partial(Text, font_size=10, stroke_width=1.5, fill_opacity=1, z_index=1)

NoteCircle = partial(Circle, fill_opacity=1, z_index=1)

RectangleLine = partial(Line, stroke_width=7, color=COLOR_RED, stroke_opacity=0.6)
SlackLine = partial(Line, stroke_width=7, color=COLOR_BLUE, stroke_opacity=0.6)


def create_note_circle(radius: float, interval: str, **kwargs):
    if interval == "1":
        color = COLOR_1
    elif interval == "3" or interval == "b3":
        color = COLOR_3
    elif interval == "5":
        color = COLOR_5
    elif interval == "7" or interval == "b7" or interval == "4":
        color = COLOR_47
    else:
        color = BLACK

    if len(interval) == 1:
        return VGroup(
            Circle(radius=radius, color=color, fill_opacity=1, z_index=1, **kwargs),
            SingleText(interval)
            ).set_z_index(1)
    else:
        return VGroup(
            Circle(radius=radius, color=color, fill_opacity=1, z_index=1, **kwargs),
            DoubleText(interval)
            ).set_z_index(1)


class NotesRectangle(VGroup):
    '''
    Should not be used.
    '''
    def __init__(self,
                 fretboard_length: float = 6,
                 fretboard_width: float = 0.3,
                 **kwargs):
        self.fretboard_length = fretboard_length
        self.fretboard_width = fretboard_width
        VGroup.__init__(self, **kwargs)

        note_1 = create_note_circle(radius=0.3*fretboard_width, interval="1")
        note_b3 = create_note_circle(radius=0.3*fretboard_width, interval="b3")\
            .move_to(note_1.get_center() + np.array((3*fretboard_length*2/24, 0, 0)))
        note_5 = create_note_circle(radius=0.3*fretboard_width, interval="5")\
            .move_to(note_1.get_center() + np.array((0, -fretboard_width, 0)))
        note_b7 = create_note_circle(radius=0.3*fretboard_width, interval="b7")\
            .move_to(note_5.get_center() + np.array((3*fretboard_length*2/24, 0, 0)))

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
    '''
    Should not be used.
    '''
    def __init__(self,
                 fretboard_length: float = 6,
                 fretboard_width: float = 0.3,
                 **kwargs):
        self.fretboard_length = fretboard_length
        self.fretboard_width = fretboard_width
        VGroup.__init__(self, **kwargs)

        note_1 = create_note_circle(radius=0.3*fretboard_width, interval="1")
        note_b3 = create_note_circle(radius=0.3*fretboard_width, interval="b3")\
            .move_to(note_1.get_center() + np.array((-2 * fretboard_length * 2 / 24, fretboard_width, 0)))
        note_5 = create_note_circle(radius=0.3*fretboard_width, interval="5")\
            .move_to(note_1.get_center() + np.array((0, -fretboard_width, 0)))
        note_4l = create_note_circle(radius=0.3*fretboard_width, interval="4")\
            .move_to(note_5.get_center() + np.array((-2*fretboard_length*2/24, 0, 0)))
        note_4h = create_note_circle(radius=0.3*fretboard_width, interval="4")\
            .move_to(note_1.get_center() + np.array((0, fretboard_width, 0)))
        note_b7 = create_note_circle(radius=0.3*fretboard_width, interval="b7")\
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