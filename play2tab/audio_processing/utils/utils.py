import numpy as np

def generate_pitches_from_midi(pitches=[69], dur=0.5, Fs=4000, amp=1, phase=0):
    """Generation of sinusoids for a given list of MIDI pitches

    Notebook: C1/C1S1_MusicalNotesPitches.ipynb

    Args:
        pitches (list): List of MIDI pitches (Default value = [69])
        dur (float): Duration (in seconds) of each sinusoid (Default value = 0.5)
        Fs (scalar): Sampling rate (Default value = 4000)
        amp (float): Amplitude of generated signal (Default value = 1)
        phase (float): Phase of sinusoid (Default value = 0)

    Returns:
        x (np.ndarray): Signal
        t (np.ndarray): Time axis (in seconds)
    """
    N = int(dur * Fs)
    t = np.arange(N) / Fs
    x = []
    for p in pitches:
        freq = 2 ** ((p - 69) / 12) * 440
        x = np.append(x, np.sin(2 * np.pi * (freq * t - phase)))
    x = amp * x / np.max(x)
    return x, t