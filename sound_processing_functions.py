import matplotlib.pyplot as plt
import numpy as np
from numpy.lib import stride_tricks
from scipy.fftpack import fft
import scipy
import matplotlib as mpl

from scipy.signal import lfilter
from scipy.signal import hamming
from segment_axis import segment_axis
from scipy.fftpack.realtransforms import dct

import mfcc_MP

mpl.rcParams['agg.path.chunksize'] = 10000

ABBRIVATIONS = {}

# features
ABBRIVATIONS["zcr"] = "Zero Crossing Rate"
ABBRIVATIONS["rms"] = "Root Mean Square"
ABBRIVATIONS["sc"] = "Spectral Centroid"
ABBRIVATIONS["sf"] = "Spectral Flux"
ABBRIVATIONS["sr"] = "Spectral Rolloff"

# aggregations
ABBRIVATIONS["var"] = "Variance"
ABBRIVATIONS["std"] = "Standard Deviation"
ABBRIVATIONS["mean"] = "Average"

PLOT_WIDTH = 15
PLOT_HEIGHT = 3.5

def show_mono_waveform(samples):
    mpl.rcParams['agg.path.chunksize'] = 10000
    fig = plt.figure(num=None, figsize=(PLOT_WIDTH, PLOT_HEIGHT), dpi=150, facecolor='w', edgecolor='k')

    channel_1 = fig.add_subplot(111)
    channel_1.set_ylabel('Channel 1')
    # channel_1.set_xlim(0,song_length) # todo
    channel_1.set_ylim(-40000, 40000)

    channel_1.plot(samples, 'k')

    plt.show()
    plt.clf()


def show_stereo_waveform(samples):
    mpl.rcParams['agg.path.chunksize'] = 10000
    fig = plt.figure(num=None, figsize=(PLOT_WIDTH, PLOT_HEIGHT*2), dpi=150, facecolor='w', edgecolor='k')

    channel_1 = fig.add_subplot(211)
    channel_1.set_ylabel('Channel 1')
    # channel_1.set_xlim(0,song_length) # todo
    channel_1.set_ylim(-40000, 40000)
    channel_1.plot(samples[:, 0], linewidth=1)

    channel_2 = fig.add_subplot(212)
    channel_2.set_ylabel('Channel 2')
    channel_2.set_xlabel('Time (s)')
    channel_2.set_ylim(-40000, 40000)
    # channel_2.set_xlim(0,song_length) # todo
    channel_2.plot(samples[:, 1], linewidth=1)

    plt.show()
    plt.clf()


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize / 2.0))), sig)
    # samples = np.append(np.zeros(np.floor(frameSize / 2.0)), sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    cols = int(cols)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    samples = samples.astype(np.int16)
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames = frames.astype(np.float64)
    frames *= win
    frames = frames.astype(np.int16)
    return np.fft.rfft(frames)


def plotstft(samples, samplerate, binsize=2 ** 10, plotpath=None, colormap="jet", ax=None, fig=None):
    s = stft(samples, binsize)
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    sshow[np.where(sshow == 0)] = 10**-10   # I put this in because in the following line there are consistent issues
    # with dividing by zeros
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    if ax is None:
        fig, ax = plt.subplots(1, 1, sharey=True, figsize=(15, 7.5), dpi=150, facecolor='w', edgecolor='k')
    # fig.set_figheight(20)
    # fig.set_figwidth(10)
    cax = ax.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_ticks([np.min(ims), 0, np.max(ims)])
    cbar.set_label('decibel', rotation=270)
    # plt.set_colorbar()

    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (hz)")
    ax.set_xlim([0, timebins - 1])
    ax.set_ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins - 1, 5))
    ax.set_xticks(xlocs, ["%.02f" % l for l in ((xlocs * len(samples) / timebins) + (0.5 * binsize)) / samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins - 1, 10)))
    ax.set_yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    plt.show()
    plt.clf()
    b = ["%.02f" % l for l in ((xlocs * len(samples) / timebins) + (0.5 * binsize)) / samplerate]
    return xlocs, b, timebins


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins - 1) / max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):int(scale[i + 1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i + 1])])]

    return newspec, freqs


def show_feature_superimposed(data, item, genre, feature_data, timestamps, label_name, squared_wf=False):
    fig = plt.figure(num=None, figsize=(PLOT_WIDTH, PLOT_HEIGHT), dpi=150, facecolor='w', edgecolor='k');
    channel_1 = fig.add_subplot(111);
    channel_1.set_ylabel('Channel 1');
    channel_1.set_xlabel('time');
    # plot waveform
    scaled_wf_x = ((np.arange(0, data[genre]["wavedata"][item].shape[0]).astype(np.float)) / data[genre][
        "samplerate"][item]) * 1000.0
    if squared_wf:
        scaled_wf_y = (data[genre]["wavedata"][item] ** 2 / np.max(data[genre]["wavedata"][item] ** 2))
    else:
        scaled_wf_y = (data[genre]["wavedata"][item] / np.max(data[genre]["wavedata"][item]) / 2.0) + 0.5

    # scaled_wf_x = scaled_wf_x**2
    plt.plot(scaled_wf_x, scaled_wf_y, color='lightgrey', label='Raw signal (scaled)');
    # plot feature-data
    scaled_fd_x = timestamps * 1000.0
    # scaled_fd_x = timestamps
    scaled_fd_y = (feature_data / np.max(feature_data))
    plt.plot(scaled_fd_x, scaled_fd_y, color='r', label=label_name);
    plt.legend()
    plt.show()
    plt.clf()


def zero_crossing_rate(wavedata, block_length, sample_rate, include_time_stamp=True):
    # how many blocks have to be processed?
    num_blocks = int(np.ceil(len(wavedata) / block_length))
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0, num_blocks - 1) * (block_length / float(sample_rate)))
    zcr = []
    for i in range(0, num_blocks - 1):
        start = i * block_length
        stop = np.min([(start + block_length - 1), len(wavedata)])
        zc = 0.5 * np.mean(np.abs(np.diff(np.sign(wavedata[start:stop]))))
        zcr.append(zc)
    if include_time_stamp:
        return np.asarray(zcr), np.asarray(timestamps)
    else:
        return np.asarray(zcr)


def plot_comparison(data, feature):
    labels = []
    means = []
    SEM = []
    for genre in data.keys():
        labels.append(genre)
        mean_temp = []
        for current in data[genre][feature]:
            mean_temp.append(np.mean(current))
        means.append(np.mean(mean_temp))
        SEM.append(scipy.stats.sem(mean_temp))
    ind = np.arange(len(data.keys()))
    # fig, ax = plt.subplots(figsize=(15, 15))
    fig, ax = plt.subplots(figsize=(15, 15), dpi=150, facecolor='w', edgecolor='k')
    rects1 = ax.bar(ind, means, 0.7, color='b', yerr=SEM)
    ax.set_xticklabels(labels, fontsize=22)
    ax.set_xticks(ind)
    ax.set_ylabel('MEAN', fontsize=22)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_title("{0} Results".format(feature), fontsize=22)
    plt.show()
    plt.clf()


def root_mean_square(wavedata, block_length, sample_rate, include_time_stamp=True):
    # how many blocks have to be processed?
    num_blocks = int(np.ceil(len(wavedata) / block_length))
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0, num_blocks - 1) * (block_length / float(sample_rate)))
    rms = []
    for i in range(0, num_blocks - 1):
        start = i * block_length
        stop = np.min([(start + block_length - 1), len(wavedata)])
        rms_seg = np.sqrt(np.mean(wavedata[start:stop] ** 2))
        rms.append(rms_seg)
    if include_time_stamp:
        return np.asarray(rms), np.asarray(timestamps)
    else:
        return np.asarray(rms)


def spectral_centroid(wavedata, window_size, sample_rate, include_time_stamp=True):
    # num_blocks = int(np.ceil(len(wavedata) / block_length))
    magnitude_spectrum = stft(wavedata, window_size)
    timebins, freqbins = np.shape(magnitude_spectrum)
    timestamps = (np.arange(0, timebins - 1) * len(wavedata)/sample_rate/timebins)
    sc = []
    for t in range(timebins - 1):
        power_spectrum = np.abs(magnitude_spectrum[t]) ** 2
        sc_t = np.sum(power_spectrum * np.arange(1, freqbins + 1)) / np.sum(power_spectrum)
        sc.append(sc_t)
    sc = np.asarray(sc)
    sc = np.nan_to_num(sc)
    if include_time_stamp:
        return sc, np.asarray(timestamps)
    else:
        return sc


def spectral_rolloff(wavedata, window_size, sample_rate, k=0.85, include_time_stamp=True):
    # convert to frequency domain
    magnitude_spectrum = stft(wavedata, window_size)
    power_spectrum = np.abs(magnitude_spectrum) ** 2
    timebins, freqbins = np.shape(magnitude_spectrum)
    timestamps = (np.arange(0, timebins - 1) * len(wavedata) / sample_rate / timebins)
    sr = []
    spectralSum = np.sum(power_spectrum, axis=1)
    for t in range(timebins - 1):
        # find frequency-bin indeces where the cummulative sum of all bins is higher
        # than k-percent of the sum of all bins. Lowest index = Rolloff
        sr_t = np.where(np.cumsum(power_spectrum[t, :]) >= k * spectralSum[t])[0][0]
        sr.append(sr_t)
    sr = np.asarray(sr).astype(float)
    # convert frequency-bin index to frequency in Hz
    sr = (sr / freqbins) * (sample_rate / 2.0)
    if include_time_stamp:
        return sr, np.asarray(timestamps)
    else:
        return sr


def spectral_flux(wavedata, window_size, sample_rate, include_time_stamp=True):
    # convert to frequency domain
    magnitude_spectrum = stft(wavedata, window_size)
    timebins, freqbins = np.shape(magnitude_spectrum)
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0, timebins - 1) * len(wavedata) / sample_rate / timebins)
    sf = np.sqrt(np.sum(np.diff(np.abs(magnitude_spectrum)) ** 2, axis=1)) / freqbins
    if include_time_stamp:
        return sf[1:], np.asarray(timestamps)
    else:
        return sf[1:]


def plot_magnitude_spectrum(mag_spect, mspect, colormap='jet'):
    fig, ax = plt.subplots(2, 1, sharey=True, figsize=(PLOT_WIDTH, 7.5), dpi=150, facecolor='w', edgecolor='k')
    cax_1 = ax[0].imshow(np.transpose(mag_spect), origin="lower", aspect="auto",  interpolation="nearest", cmap=colormap)
    ax[0].set_ylabel('Spectral bands', fontsize=22)
    ax[0].set_title('Raw amplitude magnitude spectrum averaged by chunck', fontsize=22)
    cbar_1 = fig.colorbar(cax_1, ax=ax[0])
    cbar_1.set_ticks([np.min(mag_spect), np.max(mag_spect)])
    cbar_1.set_label('Amplitude', rotation=270)
    cax_2 = ax[1].imshow(np.transpose(mspect), origin="lower", aspect="auto",  interpolation="nearest")
    ax[0].set_title('Smoothed and log transformed amplitude magnitude spectrum by chunk', fontsize=22)
    ax[1].set_ylabel('Spectral bands', fontsize=22)
    ax[1].set_xlabel('chunked bins of track', fontsize=22)
    cbar_2 = fig.colorbar(cax_2, ax=ax[1])
    cbar_2.set_ticks([np.min(mspect), np.max(mspect)])
    cbar_2.set_label('Log Amplitude', rotation=270)
    plt.show()


def plot_mfccs(mfccs, colormap='jet'):
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(PLOT_WIDTH, 4), dpi=150, facecolor='w', edgecolor='k')
    cax = ax.imshow(np.transpose(mfccs), origin="lower", aspect="auto",  interpolation="nearest", cmap=colormap)
    ax.set_ylabel('First 13 MFCCs', fontsize=22)
    ax.set_title('MFCCs averaged by chunck', fontsize=22)
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_ticks([np.min(mfccs), 0, np.max(mfccs)])
    cbar.set_label('MFCC amplitude', rotation=270)
    plt.show()


def plot_SSD_components(data):
    bark_bands = ['100', '200', '300', '400', '510', '630', '770', '920',
            '1080', '1270', '1480', '1720', '2000', '2320', '2700', '3150',
            '3700', '4400', '5300', '6400', '7700', '9500', '12000', '15500']
    ind = np.arange(len(bark_bands))
    fig, ax = plt.subplots(2, 4, figsize=(15, 15), dpi=150, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                wspace=0.4, hspace=0.2)
    fig.suptitle('Variation of Energy by band', fontsize=22)
    # mean
    ax[0][0].bar(ind, data[:, 0], 0.7, color='b')
    ax[0][0].set_xticks(np.arange(0, len(bark_bands) + 1, 1.0))
    ax[0][0].set_xticklabels(bark_bands, fontsize=6, rotation=70)
    ax[0][0].tick_params(axis='y', labelsize=10)
    ax[0][0].set_ylabel('Mean', fontsize=10)
    ax[0][0].set_title('Mean', fontsize=10)
    # Variance
    ax[0][1].bar(ind, data[:, 1], 0.7, color='b')
    ax[0][1].set_xticks(np.arange(0, len(bark_bands) + 1, 1.0))
    ax[0][1].set_xticklabels(bark_bands, fontsize=6, rotation=70)
    ax[0][1].tick_params(axis='y', labelsize=10)
    ax[0][1].set_ylabel('Variance', fontsize=10)
    ax[0][1].set_title('Variance', fontsize=10)
    # Skew
    ax[0][2].bar(ind, data[:, 2], 0.7, color='b')
    ax[0][2].set_xticks(np.arange(0, len(bark_bands) + 1, 1.0))
    ax[0][2].set_xticklabels(bark_bands, fontsize=6, rotation=70)
    ax[0][2].tick_params(axis='y', labelsize=10)
    ax[0][2].set_ylabel('Skew', fontsize=10)
    ax[0][2].set_title('Skew', fontsize=10)
    # Kurtosis
    ax[0][3].bar(ind, data[:, 3], 0.7, color='b')
    ax[0][3].set_xticks(np.arange(0, len(bark_bands) + 1, 1.0))
    ax[0][3].set_xticklabels(bark_bands, fontsize=6, rotation=70)
    ax[0][3].tick_params(axis='y', labelsize=10)
    ax[0][3].set_ylabel('Kurtosis', fontsize=10)
    ax[0][3].set_title('Kurtosis', fontsize=10)
    # Median
    ax[1][0].bar(ind, data[:, 4], 0.7, color='b')
    ax[1][0].set_xticks(np.arange(0, len(bark_bands) + 1, 1.0))
    ax[1][0].set_xticklabels(bark_bands, fontsize=6, rotation=70)
    ax[1][0].tick_params(axis='y', labelsize=10)
    ax[1][0].set_ylabel('Median', fontsize=10)
    ax[1][0].set_title('Median', fontsize=10)
    # Min
    ax[1][1].bar(ind, data[:, 5], 0.7, color='b')
    ax[1][1].set_xticks(np.arange(0, len(bark_bands) + 1, 1.0))
    ax[1][1].set_xticklabels(bark_bands, fontsize=6, rotation=70)
    ax[1][1].tick_params(axis='y', labelsize=10)
    ax[1][1].set_ylabel('Minimum Value', fontsize=10)
    ax[1][1].set_title('Min', fontsize=10)
    # Max
    ax[1][2].bar(ind, data[:, 5], 0.7, color='b')
    ax[1][2].set_xticks(np.arange(0, len(bark_bands) + 1, 1.0))
    ax[1][2].set_xticklabels(bark_bands, fontsize=6, rotation=70)
    ax[1][2].tick_params(axis='y', labelsize=10)
    ax[1][2].set_ylabel('Maximum Value', fontsize=10)
    ax[1][2].set_title('Maximum Value', fontsize=10)
    plt.show()


def periodogram(x, win, Fs=None, nfft=1024):
    if Fs == None:
        Fs = 2 * np.pi

    U = np.dot(win.conj().transpose(), win)  # compensates for the power of the window.
    Xx = fft((x * win), nfft)  # verified
    P = Xx * np.conjugate(Xx) / U

    # Compute the 1-sided or 2-sided PSD [Power/freq] or mean-square [Power].
    # Also, compute the corresponding freq vector & freq units.

    # Generate the one-sided spectrum [Power] if so wanted
    if nfft % 2 != 0:
        select = np.arange((nfft + 1) / 2)  # ODD
        P_unscaled = P[select, :]  # Take only [0,pi] or [0,pi)
        P[1:-1] = P[1:-1] * 2  # Only DC is a unique point and doesn't get doubled
    else:
        select = np.arange(nfft / 2 + 1);  # EVEN
        select = select.astype(np.int16)
        P = P[select]  # Take only [0,pi] or [0,pi) # todo remove?
        P[1:-2] = P[1:-2] * 2

    P = P / (2 * np.pi)

    return P


def plot_spectrogram(spectrogram, title_label, colormap='jet'):
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(PLOT_WIDTH, 4), dpi=150, facecolor='w', edgecolor='k')
    cax = ax.imshow(spectrogram, origin="lower", aspect="auto", interpolation="nearest", cmap=colormap)
    cbar_1 = fig.colorbar(cax, ax=ax)
    cbar_1.set_ticks([np.min(spectrogram), np.max(spectrogram)])
    cbar_1.set_label('Amplitude', rotation=270)
    plt.title(title_label, fontsize=22)
    plt.xlabel('Time')
    plt.ylabel('Spectral bands', fontsize=22)
    plt.show()


def calc_statistical_features(mat):
    result = np.zeros((mat.shape[0], 7))
    result[:, 0] = np.mean(mat, axis=1)
    result[:, 1] = np.var(mat, axis=1)
    result[:, 2] = scipy.stats.skew(mat, axis=1)
    result[:, 3] = scipy.stats.kurtosis(mat, axis=1)
    result[:, 4] = np.median(mat, axis=1)
    result[:, 5] = np.min(mat, axis=1)
    result[:, 6] = np.max(mat, axis=1)
    result = np.nan_to_num(result)
    return result


def nextpow2(num):
    n = 2
    i = 1
    while n < num:
        n *= 2
        i += 1
    return i


def plot_rhythm_historgram(r_hist_data):
    fig, ax = plt.subplots(sharey=True, figsize=(10, 10), dpi=150, facecolor='w', edgecolor='k')
    ax.bar(range(r_hist_data.shape[0]), r_hist_data)
    plt.xlabel('modulation frequency bins', fontsize=20)
    plt.title('Rhythm historgram - "rhythmic energy" per modulation frequency', fontsize=20)
    plt.show()


def get_filterbank():
    nfft = 1024
    lowfreq = 133.33
    linsc = 200 / 3.
    logsc = 1.0711703
    fs = 44100
    nlinfilt = 13
    nlogfilt = 27
    nfilt = nlinfilt + nlogfilt
    freqs = np.zeros(nfilt + 2)
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt - 1] * logsc ** np.arange(1, nlogfilt + 3)
    heights = 2. / (freqs[2:] - freqs[0:-2])
    fbank = np.zeros((nfilt, nfft))
    nfreqs = np.arange(nfft) / (1. * nfft) * fs
    for j in range(nfilt):
        low = freqs[j]
        cen = freqs[j + 1]
        hi = freqs[j + 2]
        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=np.int)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        np.floor(hi * nfft / fs) + 1, dtype=np.int)
        lslope = heights[j] / (cen - low)
        rslope = heights[j] / (hi - cen)
        fbank[j][lid] = lslope * (nfreqs[lid] - low)
        fbank[j][rid] = rslope * (hi - nfreqs[rid])
    return fbank


def get_mag_spect(wdata):
    nwin = 256
    nfft = 1024
    fs = 16000
    nceps = 13
    prefac = 0.97
    over = nwin - 160
    filtered_data = lfilter([1., -prefac], 1, wdata)
    windows = hamming(256, sym=0)
    # split up waveform into overlapping frames
    framed_data = segment_axis(filtered_data, nwin, over) * windows
    # compute fft and get amplitude magnitude over spectrum of frequencies
    mag_spect = np.abs(fft(framed_data, nfft, axis=-1))
    return mag_spect


def get_MFCCs_type_1(data_wave, samplerate, do_return_raw_mag_spect=False):
    # set number of MFCCs to compute
    nceps = 13
    filterbank = get_filterbank()
    # set chuncksize
    size_of_chunk = samplerate * 10
    num_of_chunks = data_wave.shape[0] / size_of_chunk
    wavedata_chunked = []
    # divide waveform into chunks
    for j in range(1, int(num_of_chunks), 2):
        from_ind = j*size_of_chunk
        to_ind = (j+1)*size_of_chunk
        wavedata_chunked.append(data_wave[from_ind:to_ind])
    magnitude_spectrum_chunked = []
    mspec_1_chunked = []
    MFCCs_1_chunked = []
    spec_1_chunked = []
    for chunk in wavedata_chunked:
        # compute amplitude magnitude of spectrum by chunck.  Each chunck will be split up into frames of size 1024
        magnitude_spectrum_chunked.append(get_mag_spect(chunk))
        # apply melscaling and smoothing via dot product (matrix multiplication) of mag spectrum with filterbank and
        # then apply log transformation
        mspec_1_temp = np.log10(np.dot(magnitude_spectrum_chunked[magnitude_spectrum_chunked.__len__()-1], filterbank.T))
        # apply a descrete cosine transform to get MFCCs
        MFCCs_1_temp = dct(mspec_1_temp, type=2, norm='ortho', axis=-1)[:, :nceps]
        # to reduce data needed to store in memory, get mean of MFCCs
        spec_1_chunked.append(np.mean(magnitude_spectrum_chunked[0], axis=0))
        mspec_1_chunked.append(np.mean(mspec_1_temp, axis=0)[:nceps])
        MFCCs_1_chunked.append(np.mean(MFCCs_1_temp, axis=0))
        temp_insert=0
    if do_return_raw_mag_spect:
        return MFCCs_1_chunked, mspec_1_chunked, spec_1_chunked
    else:
        return MFCCs_1_chunked, mspec_1_chunked,


def get_MFCCs_type_2(data_wave, samplerate, do_return_raw_mag_spect=False):
    nceps = 13
    # set chuncksize
    size_of_chunk = samplerate * 10
    num_of_chunks = data_wave.shape[0] / size_of_chunk
    wavedata_chunked = []
    # divide waveform into chunks
    for j in range(1, int(num_of_chunks), 2):
        from_ind = j*size_of_chunk
        to_ind = (j+1)*size_of_chunk
        wavedata_chunked.append(data_wave[from_ind:to_ind])
    mspec_2_chunked = []
    MFCCs_2_chunked = []
    spec_2_chunked = []
    for chunk in wavedata_chunked:
        # send wavefrom chunck to MP.mfcc which is at its heart the same as using MFCC_type_1 function but with a more
        # detailed computation of the filterbank.  The added spec_2_chunked returned parameter is only the raw amplitude
        # magnitude of the spectrum prior to the filtering and log transformation.
        MFCCs_2_temp, mspec_2_temp, spec_2_temp = mfcc_MP.mfcc(get_mag_spect(chunk))
        mspec_2_chunked.append(np.mean(mspec_2_temp, axis=0)[:nceps])
        MFCCs_2_chunked.append(np.mean(MFCCs_2_temp, axis=0))
        spec_2_chunked.append(np.mean(spec_2_temp, axis=0))
    if do_return_raw_mag_spect:
        return MFCCs_2_chunked, mspec_2_chunked, spec_2_chunked
    else:
        return MFCCs_2_chunked, mspec_2_chunked


def do_rythm_analysis(data_wave, samplerate, return_intermediate_steps=False):
    # various parameters for analysis
    skip_leadin_fadeout = 1
    step_width = 3
    segment_size = 2 ** 18
    fft_window_size = 1024
    # border definitions of the 24 critical bands of hearing
    bark = [100, 200, 300, 400, 510, 630, 770, 920,
            1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150,
            3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]
    n_bark_bands = len(bark)
    CONST_spread = np.zeros((n_bark_bands, n_bark_bands))
    eq_loudness = np.array(
        [[55, 40, 32, 24, 19, 14, 10, 6, 4, 3, 2,
          2, 0, -2, -5, -4, 0, 5, 10, 14, 25, 35],
         [66, 52, 43, 37, 32, 27, 23, 21, 20, 20, 20,
          20, 19, 16, 13, 13, 18, 22, 25, 30, 40, 50],
         [76, 64, 57, 51, 47, 43, 41, 41, 40, 40, 40,
          39.5, 38, 35, 33, 33, 35, 41, 46, 50, 60, 70],
         [89, 79, 74, 70, 66, 63, 61, 60, 60, 60, 60,
          59, 56, 53, 52, 53, 56, 61, 65, 70, 80, 90],
         [103, 96, 92, 88, 85, 83, 81, 80, 80, 80, 80,
          79, 76, 72, 70, 70, 75, 79, 83, 87, 95, 105],
         [118, 110, 107, 105, 103, 102, 101, 100, 100, 100, 100,
          99, 97, 94, 90, 90, 95, 100, 103, 105, 108, 115]])
    loudn_freq = np.array(
        [31.62, 50, 70.7, 100, 141.4, 200, 316.2, 500,
         707.1, 1000, 1414, 1682, 2000, 2515, 3162, 3976,
         5000, 7071, 10000, 11890, 14140, 15500])
    # calculate bark-filterbank
    loudn_bark = np.zeros((eq_loudness.shape[0], len(bark)))
    k = 0
    j = 0
    for bsi in bark:
        while j < len(loudn_freq) and bsi > loudn_freq[j]:
            j += 1
        j -= 1
        if np.where(loudn_freq == bsi)[0].size != 0:
            loudn_bark[:, k] = eq_loudness[:, np.where(loudn_freq == bsi)][:, 0, 0]
        else:
            w1 = 1 / np.abs(loudn_freq[j] - bsi)
            w2 = 1 / np.abs(loudn_freq[j + 1] - bsi)
            loudn_bark[:, k] = (eq_loudness[:, j] * w1 + eq_loudness[:, j + 1] * w2) / (w1 + w2)
        k += 1
    # Compute specific loudness sensation per critical band
    # phon-mappings
    phon = [3, 20, 40, 60, 80, 100, 101]
    idx = np.arange(fft_window_size)
    idx = idx.astype(np.int16)
    duration = data_wave.shape[0] / samplerate
    # calculate frequency values on y-axis (for bark scale calculation)
    freq_axis = float(samplerate) / fft_window_size * np.arange(0, (fft_window_size / 2) + 1)
    # modulation frequency x-axis (after 2nd fft)
    mod_freq_res = 1 / (float(segment_size) / samplerate)
    mod_freq_axis = mod_freq_res * np.arange(257)  # modulation frequencies along
    fluct_curve = 1 / (mod_freq_axis / 4 + 4 / mod_freq_axis)
    # find position of wave segment
    skip_seg = skip_leadin_fadeout
    seg_pos = np.array([1, segment_size])
    if ((skip_leadin_fadeout > 0) or (step_width > 1)):
        if (duration < 45):
            step_width = 1
            skip_seg = 0
        else:
            seg_pos = seg_pos + segment_size * skip_seg;
    # extract wave segment that will be processed
    wavsegment = data_wave[seg_pos[0] - 1:seg_pos[1]]
    # adjust hearing threshold
    wavsegment = 0.0875 * wavsegment * (2 ** 15)
    # spectrogram: real FFT with hanning window and 50 % overlap
    # number of iterations with 50% overlap
    n_iter = wavsegment.shape[0] / fft_window_size * 2 - 1
    n_iter = int(n_iter)
    w = np.hanning(fft_window_size)
    spectrogr = np.zeros((int(fft_window_size / 2 + 1), n_iter))
    # stepping through the wave segment,
    # building spectrum for each window
    for j in range(n_iter):
        spectrogr[:, j] = periodogram(x=wavsegment[idx], win=w)
        idx = idx + fft_window_size / 2
        idx = idx.astype(np.int16)
    Pxx = spectrogr
    # Apply Bark-Filter
    matrix_bark = np.zeros((len(bark), Pxx.shape[1]))
    barks = bark[:]
    barks.insert(0, 0)
    for j in range(len(barks) - 1):
        matrix_bark[j] = np.sum(Pxx[((freq_axis >= barks[j]) & (freq_axis < barks[j + 1]))], axis=0)
    # Spectral Masking
    # SPREADING FUNCTION FOR SPECTRAL MASKING
    # CONST_spread contains matrix of spectral frequency masking factors
    for j in range(n_bark_bands):
        CONST_spread[j, :] = 10 ** ((15.81 + 7.5 * ((j - np.arange(n_bark_bands)) + 0.474) - 17.5 * (
            1 + ((j - np.arange(n_bark_bands)) + 0.474) ** 2) ** 0.5) / 10)
    spread = CONST_spread[0:matrix_bark.shape[0], :]
    matrix_spec_masked = np.dot(spread, matrix_bark)
    # Map to Decibel Scale
    matrix_decibel = matrix_spec_masked
    matrix_decibel[np.where(matrix_decibel < 1)] = 1
    matrix_decibel = 10 * np.log10(matrix_decibel)
    # Transform to Phon Scale
    n_bands = matrix_decibel.shape[0]
    t = matrix_decibel.shape[1]
    table_dim = n_bands
    cbv = np.concatenate((np.tile(np.inf, (table_dim, 1)), loudn_bark[:, 0:n_bands].transpose()), 1)
    phons = phon[:]
    phons.insert(0, 0)
    phons = np.asarray(phons)
    # init lowest level = 2
    levels = np.tile(2, (n_bands, t))
    for lev in range(1, 6):
        db_thislev = np.tile(np.asarray([cbv[:, lev]]).transpose(), (1, t))
        levels[np.where(matrix_decibel > db_thislev)] = lev + 2
    # the matrix 'levels' stores the correct Phon level for each datapoint
    cbv_ind_hi = np.ravel_multi_index(dims=(table_dim, 7), multi_index=np.array(
        [np.tile(np.array([range(0, table_dim)]).transpose(), (1, t)), levels - 1]), order='F')
    cbv_ind_lo = np.ravel_multi_index(dims=(table_dim, 7), multi_index=np.array(
        [np.tile(np.array([range(0, table_dim)]).transpose(), (1, t)), levels - 2]), order='F')
    # interpolation factor % OPT: pre-calc diff
    ifac = (matrix_decibel[:, 0:t] - cbv.transpose().ravel()[cbv_ind_lo]) / (cbv.transpose().ravel()[cbv_ind_hi] - cbv.transpose().ravel()[cbv_ind_lo])
    # keeps the upper phon value;
    ifac[np.where(levels == 2)] = 1
    # keeps the upper phon value;
    ifac[np.where(levels == 8)] = 1
    matrix_phon = matrix_decibel
    matrix_phon[:, 0:t] = phons.transpose().ravel()[levels - 2] + (ifac * (phons.transpose().ravel()[levels - 1] - phons.transpose().ravel()[levels - 2]))  # OPT: pre-calc diff
    # Transform to Sone Scale
    idx = np.where(matrix_phon >= 40)
    not_idx = np.where(matrix_phon < 40)
    matrix_sone = matrix_phon
    matrix_sone[idx] = 2 ** ((matrix_sone[idx] - 40) / 10)
    matrix_sone[not_idx] = (matrix_sone[not_idx] / 40) ** 2.642
    # compute Statistical Spectrum Descriptors
    ssd = calc_statistical_features(matrix_sone)
    # calculate Rhythm Patterns
    fft_size = 2 ** (nextpow2(matrix_sone.shape[1]))
    rhythm_patterns = np.zeros((matrix_sone.shape[0], fft_size), dtype=np.complex128)
    # calculate fourier transform for each bark scale
    for b in range(0, matrix_sone.shape[0]):
        rhythm_patterns[b, :] = fft(matrix_sone[b, :], fft_size)
    # normalize results
    rhythm_patterns = rhythm_patterns / 256
    # take first 60 values of fft result including DC component
    feature_part_xaxis_rp = range(0, 60)
    rp = np.abs(rhythm_patterns[:, feature_part_xaxis_rp])
    rh = np.sum(np.abs(rhythm_patterns[:, feature_part_xaxis_rp]), axis=0)
    if return_intermediate_steps:
        return ssd, rh, rp, spectrogr, matrix_bark, matrix_spec_masked, matrix_decibel, matrix_phon, matrix_sone
    else:
        return ssd, rh, rp


def plot_cross_group_MFCCs(data, feature):
    labels = []
    means = []
    SEM = []
    for genre in data.keys():
        labels.append(genre)
        for i in range(0, len(data[genre][feature])):
            data[genre][feature][i][np.isinf(data[genre][feature][i])] = np.nan
        means.append(np.nanmean(data[genre][feature], axis=0))
        SEM.append(scipy.stats.sem(data[genre][feature], axis=0, nan_policy='omit'))
    means = np.array(means)
    SEM = np.array(SEM)
    ind = np.arange(len(data.keys()))
    fig, ax = plt.subplots(4, 4, figsize=(15, 15), dpi=150, facecolor='w', edgecolor='k')
    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                wspace=0.4, hspace=0.2)
    fig.suptitle('MFCC amplitude by coefficient:'+ feature, fontsize=22)
    for i in range(4):
        for j in range(4):
            if i*4+j>12:
                break
            else:
                ax[i][j].bar(ind, means[:, i*4+j], 0.7, color='b', yerr=SEM[:, i*4+j])
                ax[i][j].set_xticks(ind)
                ax[i][j].set_xticklabels(labels, fontsize=6)
                ax[i][j].tick_params(axis='y', labelsize=10)
                ax[i][j].set_ylabel('Amp', fontsize=10)
                ax[i][j].set_title('Coeficient num: ' + str(i*4+j+1), fontsize=10)
    plt.show()
    plt.clf()
    # fig, ax = plt.subplots(figsize=(15, 15))
    # fig, ax = plt.subplots(figsize=(15, 15), dpi=150, facecolor='w', edgecolor='k')
    # rects1 = ax.bar(ind, means, 0.7, color='b', yerr=SEM)
    # ax.set_xticklabels(labels, fontsize=22)
    # ax.set_xticks(ind)
    # ax.set_ylabel('MEAN', fontsize=22)
    # ax.tick_params(axis='y', labelsize=20)
    # ax.set_title("{0} Results".format(feature), fontsize=22)
    # plt.show()



