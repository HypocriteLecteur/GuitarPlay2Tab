from manim import *
from guitarpro import parse, BeatStatus, NoteType, SlideType

from fretboard.fretboard import FretBoard

from visualization.animation_script import demonstrate_scale_structure
from tab_processing.tab_processing import get_beats_iter,\
    adjust_duration, extract_durations
from tab_processing.chop_tab import chop_tab
from visualization.building_blocks import slide_animation, \
    grace_animation, bend_animation, normal_animation
from utils.utils import get_two_iters

import numpy as np
from utils.color import *
import glob

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

            if prev_beat is not None and now_beat.notes[0].type == NoteType.tie:
                    now_beat.notes[0].value = prev_beat.notes[0].value  # tied beats share the same fret

            if now_beat.notes[0].effect.slides:
                if now_beat.notes[0].effect.slides[0] != SlideType.legatoSlideTo:
                    raise NotImplementedError
                slide_animation(self, fb, now_beat, future_beat, duration_sec)
            elif now_beat.notes[0].effect.grace:
                grace_animation(self, fb, now_beat, duration_sec, config['frame_rate'])
            elif now_beat.notes[0].effect.bend:
                bend_animation(scene, fb, now_beat, duration_sec)
            else:
                normal_animation(scene, fb, now_beat, duration_sec)
            prev_beat = now_beat


with tempconfig({"quality": "low_quality", "disable_caching": True, 'frame_rate': 30, 'preview': True}):
    scene = Animation()
    scene.render()