from os import listdir
import collections
import sound_processing_functions as sndprcfunc
import numpy as np
import pickle


from scipy.io import wavfile
from scipy.fftpack import fft
import mfcc_MP
from scipy.fftpack.realtransforms import dct

# some functions used in script
def save_dataframe_as_pickle(frame_to_save, save_name):
    with open(save_name, 'wb') as f:
        pickle.dump(frame_to_save, f)


def open_dataframe_pickle(name_of_pickle):
    with open(name_of_pickle, 'rb') as f:
        df_from_pickle = pickle.load(f)
    return df_from_pickle





# Some options as what kind of processing to do; set to 1 to do that option
CREATE_MASTER_LIST = 0
SAVE_MASTER_LIST = 0
DO_ZCR_ANALYSIS = 1
DO_RMS_ANALYSIS = 1
DO_SPEC_CENTR_ANALYSIS = 1
DO_SPEC_ROLL_ANALYSIS = 1
DO_SPEC_FLUX_ANALYSIS = 1
DO_MFCC_ANALYSIS = 1
DO_RYTHM_ANALYSIS = 1

# name of the various files to store track data, including some temporary files in case something happens during the
# analysis so that you don't loose all the processing you did.
savefile_master_tracklist = 'tracks_set_3.pickle'
savefile_master_ZCR_temp = 'tracks_set_3_ZCR_master_progress_save.pickle'
savefile_master_ZCR_final = 'tracks_set_3_ZCR_final.pickle'
savefile_master_RMS_temp = 'tracks_set_3_RMS_master_progress_save.pickle'
savefile_master_RMS_final = 'tracks_set_3_RMS_final.pickle'
savefile_master_SPEC_CENTR_temp = 'tracks_set_3_SPEC_CENTR_master_progress_save.pickle'
savefile_master_SPEC_CENTR_final = 'tracks_set_3_SPEC_CENTR_final.pickle'
savefile_master_SPEC_ROLL_temp = 'tracks_set_3_SPEC_ROLL_master_progress_save.pickle'
savefile_master_SPEC_ROLL_final = 'tracks_set_3_SPEC_ROLL_final.pickle'
savefile_master_SPEC_FLUX_temp = 'tracks_set_3_SPEC_FLUX_master_progress_save.pickle'
savefile_master_SPEC_FLUX_final = 'tracks_set_3_SPEC_FLUX_final.pickle'
savefile_master_MFCC_temp = 'tracks_set_3_MFCC_master_progress_save.pickle'
savefile_master_MFCC_final = 'tracks_set_3_MFCC_final.pickle'
savefile_master_RYTHM_temp = 'tracks_set_3_RYTHM_master_progress_save.pickle'
savefile_master_RYTHM_final = 'tracks_set_3_RYTHM_final.pickle'

# some constants
ZCR_block_size = 2048
RMS_block_size = 2048
SPEC_CENTR_block_size = 1024
SPEC_ROLL_block_size = 1024
SPEC_FLUX_block_size = 1024


if CREATE_MASTER_LIST == 1:
    # Select folder to analyze
    # dnb_tracks_WAV = '/home/nazgul/temp/Music_WAV/DnB/'
    # filenames_wav_dnb = [dnb_tracks_WAV + f for f in listdir(dnb_tracks_WAV)]
    #
    #
    # breaks_tracks_WAV = '/home/nazgul/temp/Music_WAV/Breaks/'
    # filenames_wav_breaks = [breaks_tracks_WAV + f for f in listdir(breaks_tracks_WAV)]
    #
    #
    # house_tracks_WAV = '/home/nazgul/temp/Music_WAV/House/'
    # filenames_wav_house = [house_tracks_WAV + f for f in listdir(house_tracks_WAV)]
    #
    #
    # hiphop_tracks_WAV = '/home/nazgul/temp/Music_WAV/Hip_Hop/'
    # filenames_wav_hiphop = [hiphop_tracks_WAV + f for f in listdir(hiphop_tracks_WAV)]
    #
    # filenames_wav_all = filenames_wav_dnb + filenames_wav_breaks + filenames_wav_house + filenames_wav_hiphop
    # labels_all = (['dnb'] * len(filenames_wav_dnb)) + (['breaks'] * len(filenames_wav_breaks)) + (['house'] * len(filenames_wav_house)) + (['Hip_Hop'] * len(filenames_wav_hiphop))


    dnb_jungle_tracks_WAV = '/home/nazgul/temp/Music_WAV/DnB_jungle/'
    filenames_wav_dnb_jungle = [dnb_jungle_tracks_WAV + f for f in listdir(dnb_jungle_tracks_WAV)]

    dnb_big_tracks_WAV = '/home/nazgul/temp/Music_WAV/DnB_big/'
    filenames_wav_dnb_big = [dnb_big_tracks_WAV + f for f in listdir(dnb_big_tracks_WAV)]

    DnB_chill_tracks_WAV = '/home/nazgul/temp/Music_WAV/DnB_chill/'
    filenames_wav_dnb_chill = [DnB_chill_tracks_WAV + f for f in listdir(DnB_chill_tracks_WAV)]

    filenames_wav_all = filenames_wav_dnb_jungle + filenames_wav_dnb_big + filenames_wav_dnb_chill
    labels_all = (['dnb_jungle'] * len(filenames_wav_dnb_jungle)) + (['dnb_big'] * len(filenames_wav_dnb_big)) + (
    ['dnb_chill'] * len(filenames_wav_dnb_chill))

    if SAVE_MASTER_LIST == 1:
        save_dataframe_as_pickle([filenames_wav_all, labels_all], savefile_master_tracklist)

else:
    filenames_wav_all, labels_all = open_dataframe_pickle(savefile_master_tracklist)

# Zero-point crossing rate - features regarding noise levels in the tracks
if DO_ZCR_ANALYSIS == 1:
    # initialize music collection
    sound_files = collections.defaultdict(dict)
    ZCR = collections.defaultdict(list)
    ZCR['path'] = filenames_wav_all
    ZCR['label'] = labels_all
    # load sound files
    samplerate_master = []
    wavedata_master = []
    number_of_samples_master = []
    ZCR_master = []
    for i, track_path in enumerate(ZCR['path']):
        try:
            print('starting: ', track_path, '  track number: ', i, ' of ', len(ZCR['path']))
            samplerate, wavedata = wavfile.read(track_path)
            samplerate_master.append(samplerate)
            # wavedata_master.append(wavedata)
            number_of_samples_master.append(wavedata.shape[0])
            # audio preprocessing
            if wavedata.shape[1] > 1:
                # use combine the channels by calculating their geometric mean
                wavedata = np.mean(wavedata, axis=1)
            ZCR_temp, _ = sndprcfunc.zero_crossing_rate(wavedata, ZCR_block_size, samplerate_master[i])
            ZCR_master.append(ZCR_temp)
            save_dataframe_as_pickle([ZCR_master, track_path, i], savefile_master_ZCR_temp)
        except:
            print('ERROR processing track: ', track_path, 'track number: ', i)
            temp_insert = 0
    ZCR['data'] = ZCR_master
    ZCR['sample_rate'] = samplerate_master
    ZCR['block_size'] = ZCR_block_size
    ZCR['num_of_samples'] = number_of_samples_master
    save_dataframe_as_pickle(ZCR, savefile_master_ZCR_final)
    del ZCR

# features regarding the loudness of the tracks
if DO_RMS_ANALYSIS == 1:
    # initialize music collection
    sound_files = collections.defaultdict(dict)
    RMS = collections.defaultdict(list)
    RMS['path'] = filenames_wav_all
    RMS['label'] = labels_all
    # load sound files
    samplerate_master = []
    wavedata_master = []
    number_of_samples_master = []
    RMS_master = []
    for i, track_path in enumerate(RMS['path']):
        try:
            print('starting: ', track_path, '  track number: ', i, ' of ', len(RMS['path']))
            samplerate, wavedata = wavfile.read(track_path)
            samplerate_master.append(samplerate)
            # wavedata_master.append(wavedata)
            number_of_samples_master.append(wavedata.shape[0])
            # audio preprocessing
            if wavedata.shape[1] > 1:
                # use combine the channels by calculating their geometric mean
                wavedata = np.mean(wavedata, axis=1)
            RMS_temp, _ = sndprcfunc.root_mean_square(wavedata, RMS_block_size, samplerate_master[i])
            RMS_master.append(RMS_temp)
            save_dataframe_as_pickle([RMS_master, track_path, i], savefile_master_RMS_temp)
        except:
            print('ERROR processing track: ', track_path, 'track number: ', i)
            temp_insert = 0
    RMS['data'] = RMS_master
    RMS['sample_rate'] = samplerate_master
    RMS['block_size'] = RMS_block_size
    RMS['num_of_samples'] = number_of_samples_master
    save_dataframe_as_pickle(RMS, savefile_master_RMS_final)
    del RMS

# features regarding center of gravity (balancing point of the spectrum); gives an indication of how “dark” or “bright”
# a sound is
if DO_SPEC_CENTR_ANALYSIS == 1:
    # initialize music collection
    sound_files = collections.defaultdict(dict)
    SPEC_CENTR = collections.defaultdict(list)
    SPEC_CENTR['path'] = filenames_wav_all
    SPEC_CENTR['label'] = labels_all
    # load sound files
    samplerate_master = []
    wavedata_master = []
    number_of_samples_master = []
    SPEC_CENTR_master = []
    for i, track_path in enumerate(SPEC_CENTR['path']):
        try:
            print('starting: ', track_path, '  track number: ', i, ' of ', len(SPEC_CENTR['path']))
            samplerate, wavedata = wavfile.read(track_path)
            samplerate_master.append(samplerate)
            # wavedata_master.append(wavedata)
            number_of_samples_master.append(wavedata.shape[0])
            # audio preprocessing
            if wavedata.shape[1] > 1:
                # use combine the channels by calculating their geometric mean
                wavedata = np.mean(wavedata, axis=1)
            SPEC_CENTR_temp, _ = sndprcfunc.spectral_centroid(wavedata, SPEC_CENTR_block_size, samplerate_master[i])
            SPEC_CENTR_master.append(SPEC_CENTR_temp)
            save_dataframe_as_pickle([SPEC_CENTR_master, track_path, i], savefile_master_SPEC_CENTR_temp)
        except:
            print('ERROR processing track: ', track_path, 'track number: ', i)
            temp_insert = 0
    SPEC_CENTR['data'] = SPEC_CENTR_master
    SPEC_CENTR['sample_rate'] = samplerate_master
    SPEC_CENTR['block_size'] = SPEC_CENTR_block_size
    SPEC_CENTR['num_of_samples'] = number_of_samples_master
    save_dataframe_as_pickle(SPEC_CENTR, savefile_master_SPEC_CENTR_final)
    del SPEC_CENTR

# features measuring the skewness of the spectral shape;  indication of how much energy is in the lower frequencies
if DO_SPEC_ROLL_ANALYSIS == 1:
    # initialize music collection
    sound_files = collections.defaultdict(dict)
    SPEC_ROLL = collections.defaultdict(list)
    SPEC_ROLL['path'] = filenames_wav_all
    SPEC_ROLL['label'] = labels_all
    # load sound files
    samplerate_master = []
    wavedata_master = []
    number_of_samples_master = []
    SPEC_ROLL_master = []
    for i, track_path in enumerate(SPEC_ROLL['path']):
        try:
            print('starting: ', track_path, '  track number: ', i, ' of ', len(SPEC_ROLL['path']))
            samplerate, wavedata = wavfile.read(track_path)
            samplerate_master.append(samplerate)
            # wavedata_master.append(wavedata)
            number_of_samples_master.append(wavedata.shape[0])
            # audio preprocessing
            if wavedata.shape[1] > 1:
                # use combine the channels by calculating their geometric mean
                wavedata = np.mean(wavedata, axis=1)
            SPEC_ROLL_temp, _ = sndprcfunc.spectral_rolloff(wavedata, SPEC_ROLL_block_size, samplerate_master[i])
            SPEC_ROLL_master.append(SPEC_ROLL_temp)
            save_dataframe_as_pickle([SPEC_ROLL_master, track_path, i], savefile_master_SPEC_ROLL_temp)
        except:
            print('ERROR processing track: ', track_path, 'track number: ', i)
            temp_insert = 0
    SPEC_ROLL['data'] = SPEC_ROLL_master
    SPEC_ROLL['sample_rate'] = samplerate_master
    SPEC_ROLL['block_size'] = SPEC_ROLL_block_size
    SPEC_ROLL['num_of_samples'] = number_of_samples_master
    save_dataframe_as_pickle(SPEC_ROLL, savefile_master_SPEC_ROLL_final)
    del SPEC_ROLL

# features measuring the rate of local change in the spectrum
if DO_SPEC_FLUX_ANALYSIS == 1:
    # initialize music collection
    sound_files = collections.defaultdict(dict)
    SPEC_FLUX = collections.defaultdict(list)
    SPEC_FLUX['path'] = filenames_wav_all
    SPEC_FLUX['label'] = labels_all
    # load sound files
    samplerate_master = []
    wavedata_master = []
    number_of_samples_master = []
    SPEC_FLUX_master = []
    for i, track_path in enumerate(SPEC_FLUX['path']):
        try:
            print('starting: ', track_path, '  track number: ', i, ' of ', len(SPEC_FLUX['path']))
            samplerate, wavedata = wavfile.read(track_path)
            samplerate_master.append(samplerate)
            # wavedata_master.append(wavedata)
            number_of_samples_master.append(wavedata.shape[0])
            # audio preprocessing
            if wavedata.shape[1] > 1:
                # use combine the channels by calculating their geometric mean
                wavedata = np.mean(wavedata, axis=1)
            SPEC_FLUX_temp, _ = sndprcfunc.spectral_flux(wavedata, SPEC_FLUX_block_size, samplerate_master[i])
            SPEC_FLUX_master.append(SPEC_FLUX_temp)
            save_dataframe_as_pickle([SPEC_FLUX_master, track_path, i], savefile_master_SPEC_FLUX_temp)
        except:
            print('ERROR processing track: ', track_path, 'track number: ', i)
            temp_insert = 0
    SPEC_FLUX['data'] = SPEC_FLUX_master
    SPEC_FLUX['sample_rate'] = samplerate_master
    SPEC_FLUX['block_size'] = SPEC_FLUX_block_size
    SPEC_FLUX['num_of_samples'] = number_of_samples_master
    save_dataframe_as_pickle(SPEC_FLUX, savefile_master_SPEC_FLUX_final)
    del SPEC_FLUX


# other potential spectral features include:
# spectral variability -standard deviation of the bin values of the magnitude spectrum; provides an indication of how
# flat the spectrum is and if some frequency regions are much more prominent than others
# Strongest Partial - center frequency of the bin of the magnitude or power spectrum with the greatest strength; can
# provide a primitive form of pitch tracking

# Psychoaccoustical Features
# Mel-Frequency Cepstral Coefficients (MFCC); Mel (melody) scale (human auditory responsiveness); Fourier transform
# (FFT) of the decibel spectrum as if it were a signal; show rate of change in the different spectrum bands good timbre
# feature; MFCCs are the amplitudes of the resulting spectrum
if DO_MFCC_ANALYSIS == 1:
    sound_files = collections.defaultdict(dict)
    MFCC = collections.defaultdict(list)
    MFCC['path'] = filenames_wav_all
    MFCC['label'] = labels_all
    samplerate_master = []
    wavedata_master = []
    number_of_samples_master = []
    MFCC_master_mean_1 = []
    mspec_master_mean_1 = []
    MFCC_master_std_1 = []
    mspec_master_std_1 = []
    MFCC_master_mean_2 = []
    mspec_master_mean_2 = []
    spec_master_mean_2 = []
    MFCC_master_std_2 = []
    mspec_master_std_2 = []
    spec_master_std_2 = []
    filterbank = get_filterbank()
    for i, track_path in enumerate(MFCC['path']):
        try:
            print('starting: ', track_path, '  track number: ', i, ' of ', len(MFCC['path']))
            nceps = 13
            samplerate, wavedata = wavfile.read(track_path)
            samplerate_master.append(samplerate)
            number_of_samples_master.append(wavedata.shape[0])
            if wavedata.shape[1] > 1:
                # use combine the channels by calculating their geometric mean
                wavedata = np.mean(wavedata, axis=1)
            size_of_chunk = samplerate * 10
            num_of_chunks = wavedata.shape[0] / size_of_chunk
            wavedata_chunked = []
            for j in range(1, int(num_of_chunks), 2):
                wavedata_chunked.append(wavedata[j * size_of_chunk:(j+1) * size_of_chunk])
            del wavedata
            magnitude_spectrum_chunked = []
            mspec_1_chunked = []
            MFCCs_1_chunked = []
            mspec_2_chunked = []
            MFCCs_2_chunked = []
            spec_2_chunked = []
            for chunk in wavedata_chunked:
                magnitude_spectrum_chunked.append(sndprcfunc.get_mag_spect(chunk))
                mspec_1_temp = np.log10(np.dot(magnitude_spectrum_chunked[magnitude_spectrum_chunked.__len__()-1], sndprcfunc.filterbank.T))
                MFCCs_1_temp = dct(mspec_1_temp, type=2, norm='ortho', axis=-1)[:, :nceps]
                MFCCs_2_temp, mspec_2_temp, spec_2_temp = mfcc_MP.mfcc(get_mag_spect(chunk))
                mspec_1_chunked.append(np.mean(mspec_1_temp, axis=0)[:nceps])
                MFCCs_1_chunked.append(np.mean(MFCCs_1_temp, axis=0))
                mspec_2_chunked.append(np.mean(mspec_2_temp, axis=0)[:nceps])
                MFCCs_2_chunked.append(np.mean(MFCCs_2_temp, axis=0))
                spec_2_chunked.append(np.mean(spec_2_temp, axis=0))
                temp_insert = 0
            del MFCCs_2_temp, mspec_2_temp, spec_2_temp, MFCCs_1_temp, mspec_1_temp
            MFCC_master_mean_1.append(np.mean(MFCCs_1_chunked, axis=0))
            mspec_master_mean_1.append(np.mean(mspec_1_chunked, axis=0))
            MFCC_master_std_1.append(np.std(MFCCs_1_chunked, axis=0))
            mspec_master_std_1.append(np.std(mspec_1_chunked, axis=0))
            MFCC_master_mean_2.append(np.mean(MFCCs_2_chunked, axis=0))
            mspec_master_mean_2.append(np.mean(mspec_2_chunked, axis=0))
            spec_master_mean_2.append(np.mean(spec_2_chunked, axis=0))
            MFCC_master_std_2.append(np.std(MFCCs_2_chunked, axis=0))
            mspec_master_std_2.append(np.std(mspec_2_chunked, axis=0))
            spec_master_std_2.append(np.std(spec_2_chunked, axis=0))
            (save_dataframe_as_pickle([MFCC_master_mean_1, mspec_master_mean_1, MFCC_master_std_1, mspec_master_std_1,
                                       MFCC_master_mean_2, mspec_master_mean_2, spec_master_mean_2, MFCC_master_std_2,
                                       mspec_master_std_2, spec_master_std_2, track_path, i], savefile_master_MFCC_temp))
        except:
            print('ERROR processing track: ', track_path, 'track number: ', i)
    MFCC['MFCCs_mean_1'] = MFCC_master_mean_1
    MFCC['MFCCs_mean_2'] = MFCC_master_mean_2
    MFCC['mspec_mean_1'] = mspec_master_mean_1
    MFCC['mspec_mean_2'] = mspec_master_mean_2
    MFCC['spec_mean_2'] = spec_master_mean_2
    MFCC['MFCCs_std_1'] = MFCC_master_std_1
    MFCC['MFCCs_std_2'] = MFCC_master_std_2
    MFCC['mspec_std_1'] = mspec_master_std_1
    MFCC['mspec_std_2'] = mspec_master_std_2
    MFCC['spec_std_2'] = spec_master_std_2
    MFCC['sample_rate'] = samplerate_master
    MFCC['block_size'] = SPEC_FLUX_block_size
    MFCC['num_of_samples'] = number_of_samples_master
    save_dataframe_as_pickle(MFCC, savefile_master_MFCC_final)
    del MFCC


if DO_RYTHM_ANALYSIS == 1:
    sound_files = collections.defaultdict(dict)
    RYTHM = collections.defaultdict(list)
    RYTHM['path'] = filenames_wav_all
    RYTHM['label'] = labels_all
    samplerate_master = []
    wavedata_master = []
    number_of_samples_master = []
    skip_leadin_fadeout = 1
    step_width = 3
    segment_size = 2 ** 18
    fft_window_size = 1024
    bark = [100, 200, 300, 400, 510, 630, 770, 920,
            1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150,
            3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]
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
    loudn_bark = np.zeros((eq_loudness.shape[0], len(bark)))
    phon = [3, 20, 40, 60, 80, 100, 101]
    RP_master = []
    RH_master = []
    SSD_master = []
    for i, track_path in enumerate(RYTHM['path']):
        # try:
            print('starting: ', track_path, '  track number: ', i, ' of ', len(RYTHM['path']))
            idx = np.arange(fft_window_size)
            idx = idx.astype(np.int16)
            samplerate, wavedata = wavfile.read(track_path)
            samplerate_master.append(samplerate)
            number_of_samples_master.append(wavedata.shape[0])
            if wavedata.shape[1] > 1:
                # use combine the channels by calculating their geometric mean
                wavedata = np.mean(wavedata, axis=1)
            duration = wavedata.shape[0] / samplerate
            freq_axis = float(samplerate) / fft_window_size * np.arange(0, (fft_window_size / 2) + 1)
            mod_freq_res = 1 / (float(segment_size) / samplerate)
            mod_freq_axis = mod_freq_res * np.arange(257)  # modulation frequencies along
            fluct_curve = 1 / (mod_freq_axis / 4 + 4 / mod_freq_axis)
            skip_seg = skip_leadin_fadeout
            seg_pos = np.array([1, segment_size])
            if ((skip_leadin_fadeout > 0) or (step_width > 1)):
                if (duration < 45):
                    step_width = 1
                    skip_seg = 0
                else:
                    seg_pos = seg_pos + segment_size * skip_seg;
            wavsegment = wavedata[seg_pos[0] - 1:seg_pos[1]]
            wavsegment = 0.0875 * wavsegment * (2 ** 15)
            n_iter = wavsegment.shape[0] / fft_window_size * 2 - 1
            n_iter = int(n_iter)
            w = np.hanning(fft_window_size)
            spectrogr = np.zeros((int(fft_window_size / 2 + 1), n_iter))
            for j in range(n_iter):
                if j == 0:
                    temp_insert == 1
                spectrogr[:, j] = sndprcfunc.periodogram(x=wavsegment[idx], win=w)
                idx = idx + fft_window_size / 2
                idx = idx.astype(np.int16)
            Pxx = spectrogr
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
            # Apply Bark-Filter
            matrix = np.zeros((len(bark), Pxx.shape[1]))
            barks = bark[:]
            barks.insert(0, 0)
            for j in range(len(barks) - 1):
                matrix[j] = np.sum(Pxx[((freq_axis >= barks[j]) & (freq_axis < barks[j + 1]))], axis=0)
            n_bark_bands = len(bark)
            CONST_spread = np.zeros((n_bark_bands, n_bark_bands))
            for j in range(n_bark_bands):
                CONST_spread[j, :] = 10 ** ((15.81 + 7.5 * ((j - np.arange(n_bark_bands)) + 0.474) - 17.5 * (
                    1 + ((j - np.arange(n_bark_bands)) + 0.474) ** 2) ** 0.5) / 10)
            spread = CONST_spread[0:matrix.shape[0], :]
            matrix = np.dot(spread, matrix)
            matrix[np.where(matrix < 1)] = 1
            matrix = 10 * np.log10(matrix)
            n_bands = matrix.shape[0]
            t = matrix.shape[1]
            table_dim = n_bands
            cbv = np.concatenate((np.tile(np.inf, (table_dim, 1)), loudn_bark[:, 0:n_bands].transpose()), 1)
            phons = phon[:]
            phons.insert(0, 0)
            phons = np.asarray(phons)
            levels = np.tile(2, (n_bands, t))
            for lev in range(1, 6):
                db_thislev = np.tile(np.asarray([cbv[:, lev]]).transpose(), (1, t))
                levels[np.where(matrix > db_thislev)] = lev + 2
            spread = CONST_spread[0:matrix.shape[0], :]
            matrix = np.dot(spread, matrix)
            matrix[np.where(matrix < 1)] = 1
            matrix = 10 * np.log10(matrix)
            n_bands = matrix.shape[0]
            t = matrix.shape[1]
            table_dim = n_bands
            cbv = np.concatenate((np.tile(np.inf, (table_dim, 1)), loudn_bark[:, 0:n_bands].transpose()), 1)
            phons = phon[:]
            phons.insert(0, 0)
            phons = np.asarray(phons)
            levels = np.tile(2, (n_bands, t))
            for lev in range(1, 6):
                db_thislev = np.tile(np.asarray([cbv[:, lev]]).transpose(), (1, t))
                levels[np.where(matrix > db_thislev)] = lev + 2
            cbv_ind_hi = np.ravel_multi_index(dims=(table_dim, 7), multi_index=np.array(
                [np.tile(np.array([range(0, table_dim)]).transpose(), (1, t)), levels - 1]), order='F')
            cbv_ind_lo = np.ravel_multi_index(dims=(table_dim, 7), multi_index=np.array(
                [np.tile(np.array([range(0, table_dim)]).transpose(), (1, t)), levels - 2]), order='F')
            ifac = (matrix[:, 0:t] - cbv.transpose().ravel()[cbv_ind_lo]) / (
            cbv.transpose().ravel()[cbv_ind_hi] - cbv.transpose().ravel()[cbv_ind_lo])
            ifac[np.where(levels == 2)] = 1  # keeps the upper phon value;
            ifac[np.where(levels == 8)] = 1  # keeps the upper phon value;
            matrix[:, 0:t] = phons.transpose().ravel()[levels - 2] + (ifac * (
            phons.transpose().ravel()[levels - 1] - phons.transpose().ravel()[levels - 2]))  # OPT: pre-calc diff
            idx = np.where(matrix >= 40)
            not_idx = np.where(matrix < 40)
            matrix[idx] = 2 ** ((matrix[idx] - 40) / 10)
            matrix[not_idx] = (matrix[not_idx] / 40) ** 2.642
            fft_size = 2 ** (sndprcfunc.nextpow2(matrix.shape[1]))
            rhythm_patterns = np.zeros((matrix.shape[0], fft_size), dtype=np.complex128)
            # calculate fourier transform for each bark scale
            for b in range(0, matrix.shape[0]):
                rhythm_patterns[b, :] = fft(matrix[b, :], fft_size)
            # normalize results
            rhythm_patterns = rhythm_patterns / 256
            # take first 60 values of fft result including DC component
            feature_part_xaxis_rp = range(0, 60)
            rp = np.abs(rhythm_patterns[:, feature_part_xaxis_rp])
            rh = np.sum(np.abs(rhythm_patterns[:, feature_part_xaxis_rp]), axis=0)
            ssd = sndprcfunc.calc_statistical_features(matrix)
            RP_master.append(rp)
            RH_master.append(rh)
            SSD_master.append(ssd)
            save_dataframe_as_pickle([RP_master, RH_master, SSD_master, track_path, i], savefile_master_RYTHM_temp)
        # except:
        #     print('ERROR processing track: ', track_path, 'track number: ', i)
        #     temp_insert = 0
    RYTHM['RP'] = RP_master
    RYTHM['RH'] = RH_master
    RYTHM['SSD'] = SSD_master
    RYTHM['sample_rate'] = samplerate_master
    RYTHM['block_size'] = SPEC_FLUX_block_size
    RYTHM['num_of_samples'] = number_of_samples_master
    save_dataframe_as_pickle(RYTHM, savefile_master_RYTHM_final)
    del RYTHM
