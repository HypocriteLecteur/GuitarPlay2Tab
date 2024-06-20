import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

def estimate_pitch(segment, sr, fmin=librosa.note_to_hz('C3'), fmax=librosa.note_to_hz('C6')):
    
    # Compute autocorrelation of input segment.
    r = librosa.autocorrelate(segment)
    
    # Define lower and upper limits for the autocorrelation argmax.
    i_min = sr/fmax
    i_max = sr/fmin
    r[:int(i_min)] = 0
    r[int(i_max):] = 0
    
    # Find the location of the maximum autocorrelation.
    i = r.argmax()
    f0 = float(sr)/i
    return f0

def generate_sine(f0, sr, n_duration):
    n = np.arange(n_duration)
    return 0.2*np.sin(2*np.pi*f0*n/float(sr))

def estimate_pitch_and_generate_sine(x, onset_samples, i, sr):
    n0 = onset_samples[i]
    n1 = onset_samples[i+1]
    f0 = estimate_pitch(x[n0:n1], sr)
    return generate_sine(f0, sr, n1-n0)

def harmonic_summation(log_cqt, bins_per_octave, harmonic_degree):
    log_cqt_harmonic_sum = np.copy(log_cqt)

    R = 1200 / bins_per_octave
    harmonic_series_idxes = (1200 / R * np.log2(np.arange(harmonic_degree)+1)).astype(int)

    start_idx = 0    
    while len(harmonic_series_idxes) > 1:
        for i in range(start_idx, log_cqt_harmonic_sum.shape[0] - harmonic_series_idxes[-1]):
            log_cqt_harmonic_sum[i, :] = np.sum(log_cqt_harmonic_sum[i + harmonic_series_idxes, :], axis=0)
        start_idx = start_idx + log_cqt_harmonic_sum.shape[0] - harmonic_series_idxes[-1]
        harmonic_series_idxes = np.delete(harmonic_series_idxes, -1)

    return log_cqt_harmonic_sum

def extract_melody(log_cqt):
    melody_spectrogram = np.ones_like(log_cqt) * np.min(log_cqt)
    max_idxes = np.argmax(log_cqt, axis=0)
    _idxes = log_cqt[max_idxes, np.arange(len(max_idxes))] > 0
    max_idxes = max_idxes[_idxes]
    valid_idxes = np.arange(log_cqt.shape[1])[_idxes]
    melody_spectrogram[max_idxes, valid_idxes] = log_cqt[max_idxes, valid_idxes]
    return melody_spectrogram, valid_idxes

def spectrogram_idx_to_frequency(idx):
    return 2**((24+idx-69)/12)*440


if __name__ == '__main__':
    filename = 'test\\test.wav'
    x, sr = librosa.load(filename)

    # librosa.display.waveshow(x, sr=sr)
    # plt.show()

    hop_length = 200
    bins_per_octave = 12
    cqt = librosa.cqt(x, sr=sr, hop_length=hop_length, n_bins=12*8, bins_per_octave=bins_per_octave)
    log_cqt = librosa.amplitude_to_db(cqt)

    # harmonic summation
    harmonic_degree = 3
    log_cqt_harmonic_sum = harmonic_summation(log_cqt, bins_per_octave, harmonic_degree)

    melody_spectrogram, melody_idxes = extract_melody(log_cqt)

    f, (ax1, ax2) = plt.subplots(1, 2)
    librosa.display.specshow(log_cqt, sr=sr, x_axis='time', y_axis='cqt_note', hop_length=hop_length,
                             bins_per_octave=bins_per_octave, ax=ax1)
    librosa.display.specshow(melody_spectrogram, sr=sr, x_axis='time', y_axis='cqt_note', hop_length=hop_length,
                             bins_per_octave=bins_per_octave, ax=ax2)
    # plt.show()

    # pitches, magnitudes = librosa.piptrack(S=log_cqt, sr=sr, threshold=1,
    #                                    ref=np.mean)

    # chroma_map = librosa.filters.cq_to_chroma(cqt.shape[0])
    # chromagram = chroma_map.dot(cqt)
    # # Max-normalize each time step
    # chromagram = librosa.util.normalize(chromagram, axis=0)
    # librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time')
    # plt.show()

    hop_length = 50
    onset_samples = librosa.onset.onset_detect(y=x,
                                            sr=sr, units='samples', 
                                            hop_length=hop_length, 
                                            backtrack=True,
                                            pre_max=20,
                                            post_max=20,
                                            pre_avg=100,
                                            post_avg=100,
                                            delta=0.1,
                                            wait=0)
    onset_boundaries = np.concatenate([[0], onset_samples, [len(x)]])
    onset_times = librosa.samples_to_time(onset_boundaries, sr=sr)

    for onset_time in onset_times:
        plt.axvline(x=onset_time, color='r')
    plt.show()

    # tempo, beats = librosa.beat.beat_track(y=x, sr=sr, hop_length=200)
    # print("Tempo 1:", tempo)

    # pitch estimation
    y = np.array([])
    for i in range(len(onset_boundaries)-1):
        n0 = int(onset_boundaries[i] / 200)
        n1 = int(onset_boundaries[i+1] / 200)
        midi = np.median(np.argmax(melody_spectrogram[:, melody_idxes[np.where(np.logical_and(melody_idxes >= n0 , melody_idxes <= n1))]], axis=0))
        if not np.isnan(midi):
            y = np.concatenate((y, generate_sine(spectrogram_idx_to_frequency(midi), sr, (n1-n0)*200)))
            # print(librosa.hz_to_note(spectrogram_idx_to_frequency(midi)))

    
    # y = np.concatenate([
    #     estimate_pitch_and_generate_sine(x, onset_boundaries, i, sr=sr)
    #     for i in range(len(onset_boundaries)-1)
    # ])

    write('test/output.wav', sr, y)

    # cqt = librosa.cqt(y, sr=sr)
    # librosa.display.specshow(abs(cqt), sr=sr, x_axis='time', y_axis='cqt_note')

    # plt.plot(onset_env)
    # plt.xlim(0, len(onset_env))
