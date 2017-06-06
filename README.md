## Music Genre Classification with Supervised Learning
### The python scripts here are for implimenting:

1. MP3 to WAV file convertion in bulk.
2. extraction of MFCC information, low level audio track information, rhythmic features from tracks in bulk
3. visualization of extracted track components
4. building of a model for classifying music genres.  In this case there were two goals. Test various supervised classification models in categorizing:
    * seperate genres of electronic music having different BPM, time signatures, and beat structure.  Specifically, categorizing tracks into breakbeats, hip-hop, drum and bass, and house music
    * seperate sub-genres having similar BPM, time signatures, and beat structure.  Specifically, categorizing three types of drum and bass music: Jungle, minimal/chill, and big neurofunk.
	
### packages to install:
1. pydub (also requires installation of ffmpeg.  On linux box: apt-get install ffmpeg)
2. scipy
3. numpy
4. matplotlib (also requires installation of python3-tk. On linux box: apt-get install python3-tk) 

### Still to do
1. fix feature comparison graphs with proper STD bars
2. add colorbar for heat map on STFT graph
3. register zero pt crossing rate with waveform in graph
4. register RMS with waveform in graph
5. register spectral rolloff with waveform in graph
6. register spectral flux with waveform in graph
7. place titles in graph of spectra magnitude
8. expand graphing range of frequencies on spectrogram plots (spectrogram and STFF


### References:
1. [A great tutorial on processing music](http://www.ifs.tuwien.ac.at/~schindler/lectures/MIR_Feature_Extraction.html) - plenty of the signal processing code used in this project stems directly from this site.  However, the code on the site is written for python2.  I have modified the code to run on python3 and further to suite my needs.
