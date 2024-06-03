from manim import *
from guitarpro import Song

from fretboard.note import Note
from fretboard.building_blocks import NotesRectangle, NotesSlack
from fretboard.fretboard import FretBoard

import numpy as np
from utils.color import *

def demonstrate_scale_structure(scene: Scene, fb: FretBoard, song: Song, mode=None):
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