from manim import *

from tab_processing.tab_processing import extract_and_adjust_bend_durations, get_bend_points_iter
from utils.utils import get_two_iters

def slide_animation(scene, fb, now_beat, future_beat, run_time):
    note = fb.create_playing_note_circle((now_beat.notes[0].string, now_beat.notes[0].value))
    scene.add(note)
    scene.play(
        note.animate\
            .shift(LEFT * (now_beat.notes[0].value - future_beat.notes[0].value) * fb.fret_length),
        run_time=run_time,
        rate_functions=rate_functions.linear
    )
    scene.remove(note)
    return

def grace_animation(scene, fb, now_beat, run_time, frame_rate):
    note = fb.create_playing_note_circle((now_beat.notes[0].string, now_beat.notes[0].effect.grace.fret))
    scene.add(note)

    grace_duration_percentage = now_beat.duration.value / now_beat.notes[0].effect.grace.duration
    grace_duration_frame = round(run_time * grace_duration_percentage * frame_rate)
    scene.play(note.animate
            .shift(LEFT * (now_beat.notes[0].effect.grace.fret - now_beat.notes[0].value) * fb.fret_length),
            run_time=grace_duration_frame / frame_rate, rate_functions=rate_functions.linear)
    
    scene.wait((run_time * frame_rate - grace_duration_frame) / frame_rate)
    scene.remove(note)
    return

def bend_animation(scene, fb, now_beat, run_time):
    string_id = fb.strings_number - now_beat.notes[0].string
    fb.strings[string_id].set_opacity(0)
    fb.strings[string_id-1].set_opacity(0)

    note = fb.create_playing_note_circle((now_beat.notes[0].string, now_beat.notes[0].value))
    line = always_redraw(
        lambda: VGroup(
            Line(fb.string_fret_to_pos(now_beat.notes[0].string, 0) + RIGHT * 0.5 * fb.fret_length, 
                    note.get_center(), 
                    color=BLACK, 
                    stroke_width = fb.strings[string_id].stroke_width),
            Line(note.get_center(), 
                    fb.string_fret_to_pos(now_beat.notes[0].string, fb.frets_number) + RIGHT * 0.5 * fb.fret_length, 
                    color=BLACK, 
                    stroke_width = fb.strings[string_id].stroke_width)
        )
    )
    line2 = always_redraw(
        lambda: VGroup(
            Line(fb.string_fret_to_pos(now_beat.notes[0].string+1, 0) + RIGHT * 0.5 * fb.fret_length, 
                    np.array([note.get_center()[0], np.min((note.get_center()[1], fb.strings[string_id-1].get_center()[1])), 0]), 
                    color=BLACK, 
                    stroke_width = fb.strings[string_id-1].stroke_width),
            Line(np.array([note.get_center()[0], np.min((note.get_center()[1], fb.strings[string_id-1].get_center()[1])), 0]), 
                    fb.string_fret_to_pos(now_beat.notes[0].string+1, fb.frets_number) + RIGHT * 0.5 * fb.fret_length, 
                    color=BLACK, 
                    stroke_width = fb.strings[string_id-1].stroke_width)
        )
    )
    scene.add(note, line, line2)

    note.shift(DOWN * now_beat.notes[0].effect.bend.points[0].value/2 * fb.string_gap)

    bend_durations = extract_and_adjust_bend_durations(now_beat, run_time, config['frame_rate'])

    now_point_iter, next_point_iter = get_two_iters(get_bend_points_iter(now_beat.notes[0].effect.bend.points))
    for i, (now_point, next_point) in enumerate(zip(now_point_iter, next_point_iter)):
        if next_point is not None:
            scene.play(note.animate.shift(DOWN * (next_point.value - now_point.value)/2 * fb.string_gap),
                        run_time=bend_durations[i], rate_functions=rate_functions.linear)

    scene.remove(line, line2)
    fb.strings[string_id].set_opacity(1)
    fb.strings[string_id - 1].set_opacity(1)
    scene.remove(note)
    return

def normal_animation(scene, fb, now_beat, run_time):
    note = fb.create_playing_note_circle((now_beat.notes[0].string, now_beat.notes[0].value))
    scene.add(note)
    scene.wait(run_time, frozen_frame=True)
    scene.remove(note)
    return