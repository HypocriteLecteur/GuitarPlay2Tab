from manim import *
from guitarpro import parse, BeatStatus, NoteType, SlideType

from fretboard.fretboard import FretBoard

from visualization.animation_script import demonstrate_scale_structure
from tab_processing.tab_processing import get_beats_iter, get_bend_points_iter, adjust_duration, extract_durations
from tab_processing.chop_tab import chop_tab

import numpy as np
from itertools import tee
from utils.color import *
import glob

def get_two_iters(iter):
    now_it, next_it = tee(iter)
    next(next_it, None)
    return now_it, next_it

class Animation(Scene):
    def construct(self):
        # parameters -----------------------------------------------------
        song = parse('test\\ALONE.gp5')
        bpm = 116.6

        chop_tab('test\\tab_pages')

        num_measures_in_line = np.load('test\\tab_pages\\num_measures_in_line.npy')
        measure_next_tab = np.cumsum(num_measures_in_line[0::2] + num_measures_in_line[1::2])
        chopped_tabs_path = glob.glob('test\\tab_pages\\output\\*_*.png')

        background = ImageMobject("test\\merge.png")
        background.scale(2)
        
        # initialization--------------------------------------------------
        self.add(background)
        fb = FretBoard()
        self.add(fb)

        durations, beat_measure_map = extract_durations(song, bpm)
        durations = adjust_duration(durations, song, config['frame_rate'])

        # song structure--------------------------------------------------
        self.next_section(skip_animations=True)
        demonstrate_scale_structure(self, fb, song)

        # tab animations--------------------------------------------------
        self.next_section()

        lower_tab = ImageMobject(chopped_tabs_path[1]).to_edge(DOWN, buff=0)
        upper_tab = ImageMobject(chopped_tabs_path[0]).next_to(lower_tab, UP, buff=0)
        self.add(upper_tab)
        self.add(lower_tab)

        prev_beat = None
        now_beat_iter, next_beat_iter = get_two_iters(get_beats_iter(song))
        for beat_count, (duration_sec, now_beat) in enumerate(zip(durations, now_beat_iter)):
            if beat_count in beat_measure_map:
                measure = beat_measure_map[beat_count]
                if measure in measure_next_tab:
                    idx = np.where(measure_next_tab == measure)[0][0]

                    self.remove(upper_tab)
                    self.remove(lower_tab)
                    lower_tab = ImageMobject(chopped_tabs_path[3 + 2 * idx]).to_edge(DOWN, buff=0)
                    upper_tab = ImageMobject(chopped_tabs_path[2 + 2 * idx]).next_to(lower_tab, UP, buff=0)
                    self.add(upper_tab)
                    self.add(lower_tab)

            if beat_count < len(durations) - 1:
                future_beat = next(next_beat_iter)
            
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
                self.play(_note.animate
                        .shift(LEFT * (now_beat.notes[0].effect.grace.fret - now_beat.notes[0].value) * fb.fret_length),
                        run_time=grace_duration_frame / config['frame_rate'], rate_functions=rate_functions.linear)
                
                self.wait((duration_sec * config['frame_rate'] - grace_duration_frame) /
                                    config['frame_rate'])

            elif now_beat.notes[0].effect.bend:
                string_id = 6 - now_beat.notes[0].string

                fb.strings[string_id].set_opacity(0)
                fb.strings[string_id-1].set_opacity(0)
                _note = fb.create_playing_note_circle((now_beat.notes[0].string, now_beat.notes[0].value))

                line = always_redraw(
                    lambda: VGroup(
                        Line(fb.string_fret_to_pos(now_beat.notes[0].string, 0) + RIGHT * 0.5 * fb.fret_length, 
                             _note.get_center(), 
                             color=BLACK, 
                             stroke_width = fb.strings[string_id].stroke_width),
                        Line(_note.get_center(), 
                             fb.string_fret_to_pos(now_beat.notes[0].string, fb.frets_number) + RIGHT * 0.5 * fb.fret_length, 
                             color=BLACK, 
                             stroke_width = fb.strings[string_id].stroke_width)
                    )
                )
                line2 = always_redraw(
                    lambda: VGroup(
                        Line(fb.string_fret_to_pos(now_beat.notes[0].string+1, 0) + RIGHT * 0.5 * fb.fret_length, 
                             np.array([_note.get_center()[0], np.min((_note.get_center()[1], fb.strings[string_id-1].get_center()[1])), 0]), 
                             color=BLACK, 
                             stroke_width = fb.strings[string_id-1].stroke_width),
                        Line(np.array([_note.get_center()[0], np.min((_note.get_center()[1], fb.strings[string_id-1].get_center()[1])), 0]), 
                             fb.string_fret_to_pos(now_beat.notes[0].string+1, fb.frets_number) + RIGHT * 0.5 * fb.fret_length, 
                             color=BLACK, 
                             stroke_width = fb.strings[string_id-1].stroke_width)
                    )
                )
                self.add(_note, line, line2)

                _note.shift(DOWN * now_beat.notes[0].effect.bend.points[0].value/2 * fb.string_gap)

                _it1, _it2 = tee(get_bend_points_iter(now_beat.notes[0].effect.bend.points))
                next(_it2, None)
                _durations = []
                for now_point, next_point in zip(_it1, _it2):
                    if next_point is not None:
                        _durations.append((next_point.position - now_point.position)/12*duration_sec)
                for i, _duration in enumerate(_durations[:-1]):
                    _durations[i] = round(_duration * config['frame_rate']) / config['frame_rate']
                _durations = np.array(_durations)
                _durations[-1] = duration_sec - np.sum(_durations[:-1])

                _it1, _it2 = tee(get_bend_points_iter(now_beat.notes[0].effect.bend.points))
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