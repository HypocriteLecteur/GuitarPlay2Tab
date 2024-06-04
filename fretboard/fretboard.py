from manim import Circle, Line, VGroup
from manim import BLACK, GREY

from utils.color import *
from fretboard.note import Note
from fretboard.building_blocks import create_note_circle, RectangleLine, SlackLine

import numpy as np


class FretBoard(VGroup):
    def __init__(self,
                 strings_number: int = 6,
                 frets_number: int = 24,
                 tuning: str = 'E standard',
                 fretboard_length: float = 6,
                 string_gap: float = 0.3,
                 fretboard_color: ManimColor = BLACK,
                 **kwargs):
        self.strings_number = strings_number
        self.frets_number = frets_number
        self.fretboard_length = fretboard_length
        self.string_gap = string_gap
        self.fretboard_color = fretboard_color
        self.tuning = tuning
        self.fret_length = self.fretboard_length*2/self.frets_number
        VGroup.__init__(self, **kwargs)

        # -------------------------------------------------
        self.strings = VGroup()
        for i in range(self.strings_number):
            self.strings.add(
                Line(
                    np.array((-self.fretboard_length, i*self.string_gap, 0.0)),
                    np.array((self.fretboard_length, i*self.string_gap, 0.0)),
                    stroke_width=2.5 - i*0.15,
                    color=self.fretboard_color
                )
            )
        self.add(self.strings)

        # -------------------------------------------------
        self.fret_lines = VGroup()
        for i in range(self.frets_number + 1):
            self.fret_lines.add(
                Line(
                    np.array((-self.fretboard_length + i*self.fret_length, 0, 0.0)),
                    np.array((-self.fretboard_length + i*self.fret_length, (self.strings_number - 1)*self.string_gap, 0.0)),
                    stroke_width=2.5,
                    color=self.fretboard_color
                )
            )
        self.add(self.fret_lines)

        # -------------------------------------------------
        self.dots = VGroup()

        offset = np.array((0, 0.5*self.string_gap, 0))
        dots_pos = [(4, 3), (4, 5), (4, 7), (4, 9), (5, 12), (3, 12),
                    (4, 15), (4, 17), (4, 19), (4, 21), (5, 24), (3, 24)]
        for dot_pos in dots_pos:
            self.dots.add(
                Circle(radius=0.2 * self.string_gap, color=GREY, fill_opacity=1)
                .move_to(self.string_fret_to_pos(*dot_pos) + offset)
            )

        self.add(self.dots)

        # -------------------------------------------------
        if tuning == 'E standard':
            self.strings_offsets = np.array([5, 5, 5, 4, 5])
        else:
            raise NotImplementedError

    def string_fret_to_pos(self, string_num, fret_num):
        string_id = self.strings_number - string_num

        bottom_string_y = self.strings[0].get_center()[1]
        y = bottom_string_y + self.string_gap * string_id

        zero_fret_x = self.fret_lines[0].get_center()[0]
        x = zero_fret_x + self.fret_length*fret_num - self.fret_length / 2
        return np.array((x, y, 0))

    def note_difference(self, string_fret, note):
        string_difference = int(abs(self.strings_number - string_fret[0]))
        diff = int(np.mod(np.sum(self.strings_offsets[:string_difference]) + string_fret[1] - note.value, 12))
        if diff == 3:
            return "b3"
        elif diff == 7:
            return "5"
        elif diff == 5:
            return "4"
        elif diff == 10:
            return "b7"
        elif diff == 2:
            return "2"
        elif diff == 9:
            return "6"
        return None

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
        '''
        Cleaner implementation needed.
        '''
        if self.tuning != 'E standard':
            raise NotImplementedError

        rectangles = VGroup()
        for position in positions:
            if position[0] == self.strings_number:
                rectangles.add(RectangleLine(
                    self.string_fret_to_pos(position[0], position[1]),
                    self.string_fret_to_pos(position[0], position[1]+3)
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
                rectangle.add(RectangleLine(self.string_fret_to_pos(rectangle_positions[pair[0], 0], rectangle_positions[pair[0], 1]),
                                            self.string_fret_to_pos(rectangle_positions[pair[1], 0], rectangle_positions[pair[1], 1])))
            rectangles.add(rectangle)
        return rectangles

    def slacks_lines_from_positions(self, positions):
        '''
        Cleaner implementation needed.
        '''
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
                slack.add(SlackLine(self.string_fret_to_pos(slack_positions[pair[0], 0], slack_positions[pair[0], 1]),
                                    self.string_fret_to_pos(slack_positions[pair[1], 0], slack_positions[pair[1], 1])))
            slacks.add(slack)
        return slacks

    def note_group_from_positions(self, positions, interval):
        note_group = VGroup()
        for position in positions:
            note_group.add(
                self.create_note_circle(interval=interval, string_fret=position)
                )
        return note_group.set_z_index(1)
    
    def create_note_circle(self, interval, string_fret=None):
        if string_fret is not None:
            return create_note_circle(radius=0.3 * self.string_gap, interval=interval)\
                .move_to(self.string_fret_to_pos(string_fret[0], string_fret[1]))
        else:
            return create_note_circle(radius=0.3 * self.string_gap, interval=interval)
    
    def create_playing_note_circle(self, string_fret=None):
        return Circle(radius=0.5 * self.string_gap, color=RED, stroke_color=BLACK, fill_opacity=0.6).set_z_index(2)\
            .move_to(self.string_fret_to_pos(string_fret[0], string_fret[1]))