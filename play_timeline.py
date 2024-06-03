from manim import *
from typing import Iterable, Union

def play_timeline(scene: Scene, timeline: dict[float, Union[Iterable[Animation], Animation]]):
    """
    Plays a timeline of animations on a given scene.
    Args:
        scene (Scene): The scene to play the animations on.
        timeline (dict): A dictionary where the keys are the times at which the animations should start,
            and the values are the animations to play at that time. The values can be a single animation
            or an iterable of animations.

    Notes:
        Each animation in the timeline can have a different duration, so several animations can be
        running in parallel. If the value for a given time is an iterable, all the animations
        in the iterable are started at once (although they can end at different times depending
        on their run_time)
        The method returns when all animations have finished playing.
    Returns:
        None
    """
    previous_t = 0
    ending_time = 0
    for t, anims in sorted(timeline.items()):
        to_wait = t - previous_t
        if to_wait > 0:
            scene.wait(to_wait)
        previous_t = t
        if not isinstance(anims, Iterable):
            anims = [anims]
        for anim in anims:
            turn_animation_into_updater(anim)
            scene.add(anim.mobject)
            ending_time = max(ending_time, t + anim.run_time)
    if ending_time > t:
        scene.wait(ending_time - t)