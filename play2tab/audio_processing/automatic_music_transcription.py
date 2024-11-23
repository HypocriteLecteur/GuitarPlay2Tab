import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import configparser

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
    n0 = int(onset_samples[i]*sr)
    n1 = int(onset_samples[i+1]*sr)
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

def extract_melody(log_cqt, threshold=0):
    melody_spectrogram = np.ones_like(log_cqt) * np.min(log_cqt)
    max_idxes = np.argmax(log_cqt, axis=0)
    _idxes = log_cqt[max_idxes, np.arange(len(max_idxes))] > threshold
    max_idxes = max_idxes[_idxes]
    valid_idxes = np.arange(log_cqt.shape[1])[_idxes]
    melody_spectrogram[max_idxes, valid_idxes] = log_cqt[max_idxes, valid_idxes]
    return melody_spectrogram, valid_idxes

def spectrogram_idx_to_frequency(idx):
    return 2**((24+idx-69)/12)*440

def calculate_spectrogram(x, sr, cfg, display=False, axis=None):
    spectrogram = cfg['Spectrogram']
    cqt = librosa.cqt(x, sr=sr, hop_length=spectrogram.getint('hop_length'), 
                      fmin=librosa.note_to_hz(spectrogram['note_min']),
                      n_bins=spectrogram.getint('n_bins'), 
                      bins_per_octave=spectrogram.getint('bins_per_octave'))
    cqt_mag = librosa.magphase(cqt)[0]**2
    log_cqt = librosa.core.amplitude_to_db(cqt_mag ,ref=np.max) 

    # log_cqt = cqt_thresholded(log_cqt, spectrogram.getint('cqt_threshold'))

    if display:
        librosa.display.specshow(log_cqt, sr=sr, x_axis='time', y_axis='cqt_note', 
                                hop_length=spectrogram.getfloat('hop_length'),
                                fmin=librosa.note_to_hz(spectrogram['note_min']),
                                bins_per_octave=spectrogram.getint('bins_per_octave'), ax=axis)

    return log_cqt

def cqt_thresholded(cqt,thres):
    new_cqt=np.copy(cqt)
    new_cqt[new_cqt<thres]=-120
    return new_cqt

def calc_onset_env(cqt, sr, hop_length):
    return librosa.onset.onset_strength(S=cqt, sr=sr, aggregate=np.mean, hop_length=hop_length)

def calc_onset(cqt, sr, hop_length, pre_post_max, backtrack=True):
    onset_env=calc_onset_env(cqt, sr, hop_length)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env,
                                           sr=sr, units='frames', 
                                           hop_length=hop_length, 
                                           backtrack=backtrack,
                                           pre_max=pre_post_max,
                                           post_max=pre_post_max)
    onset_boundaries = np.concatenate([[0], onset_frames, [cqt.shape[1]]])
    onset_times = librosa.frames_to_time(onset_boundaries, sr=sr, hop_length=hop_length)
    return [onset_times, onset_boundaries, onset_env]

if __name__ == '__main__':
    filename = 'test\\guitar4.mp4'
    model_path='D:\\GitHub\\GuitarPlay2Tab\\GuitarPlay2Tab\\basic_pitch_torch\\assets\\basic_pitch_pytorch_icassp_2022.pth'
    x, sr = librosa.load(filename)

    from basic_pitch_torch.inference import predict
    model_output, midi_data, note_events = predict(filename, model_path=model_path)
    
    plt.figure()
    librosa.display.specshow(midi_data.get_piano_roll())
    plt.show()

    write('test/output.wav', sr*2, midi_data.synthesize())
    
    # print(model_output)


    # cfg = configparser.ConfigParser()
    # cfg.read('.\\play2tab\\audio_processing\\config.ini')
    # spectrogram = cfg['Spectrogram']

    # f, (ax1, ax2) = plt.subplots(1, 2)

    # log_cqt = calculate_spectrogram(x, sr, cfg, display=True, axis=ax1)

    # f0, voiced_flag, voiced_probs = librosa.pyin(x,
    #                                          sr=sr,
    #                                          fmin=librosa.note_to_hz('E2'),
    #                                          fmax=librosa.note_to_hz('E6'))
    # times = librosa.times_like(f0, sr=sr)
    # ax1.plot(times, f0, label='f0', color='cyan', linewidth=3)
    # plt.show()

    # # librosa.display.waveshow(x, sr=sr, alpha=0.5, ax=ax2)

    # # harmonic summation
    # # harmonic_degree = 3
    # # log_cqt_harmonic_sum = harmonic_summation(log_cqt, bins_per_octave, harmonic_degree)

    # # melody_spectrogram, melody_idxes = extract_melody(log_cqt, threshold=0)

    # # librosa.display.specshow(melody_spectrogram, sr=sr, x_axis='time', y_axis='cqt_note', hop_length=spectrogram.getint('hop_length'),
    # #                          bins_per_octave=spectrogram.getint('bins_per_octave'), ax=ax1)

    # # pitches, magnitudes = librosa.piptrack(S=log_cqt, sr=sr, threshold=1,
    # #                                    ref=np.mean)
    
    # plt.figure()
    # librosa.display.specshow(pitches)

    # # chroma_map = librosa.filters.cq_to_chroma(cqt.shape[0])
    # # chromagram = chroma_map.dot(cqt)
    # # # Max-normalize each time step
    # # chromagram = librosa.util.normalize(chromagram, axis=0)
    # # librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time')
    # # plt.show()

    # onsets = calc_onset(log_cqt, sr, hop_length=512, pre_post_max=spectrogram.getint('pre_post_max'))
    # onset_boundaries = onsets[1]

    # # plt.figure()
    # # librosa.display.waveshow(onsets[2], sr=sr, alpha=0.5)
    # # for onset_time in onsets[0]:
    # #     plt.axvline(x=onset_time, color='r')
    
    # plt.show()

    # # tempo, beats = librosa.beat.beat_track(y=x, sr=sr, hop_length=200)
    # # print("Tempo 1:", tempo)

    # # pitch estimation
    # y = np.concatenate([
    #     estimate_pitch_and_generate_sine(x, onsets[0], i, sr=sr)
    #     for i in range(len(onset_boundaries)-1)
    # ])

    # write('test/output.wav', sr, y)

    # # plt.plot(onset_env)
    # # plt.xlim(0, len(onset_env))
