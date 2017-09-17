from pydub import AudioSegment
from os import listdir

# dnb_tracks_MP3 = '/home/markhyphen/tracks/dnb/'
# dnb_tracks_WAV = '/home/markhyphen/tracks/dnb/'
# filenames_mp3 = [dnb_tracks_MP3 + f for f in listdir(dnb_tracks_MP3)]
# filenames_wav = [dnb_tracks_WAV + f[:-3] + 'wav' for f in listdir(dnb_tracks_MP3)]

# breaks_tracks_MP3 = '/home/nazgul/temp/music/Breakbeat/'
# breaks_tracks_WAV = '/home/nazgul/temp/Music_WAV/Breaks/'
# filenames_mp3 = [breaks_tracks_MP3 + f for f in listdir(breaks_tracks_MP3)]
# filenames_wav = [breaks_tracks_WAV + f[:-3] + 'wav' for f in listdir(breaks_tracks_MP3)]

# house_tracks_MP3 = '/home/markhyphen/tracks/house/'
# house_tracks_WAV = '/home/markhyphen/tracks/house/'
# filenames_mp3 = [house_tracks_MP3 + f for f in listdir(house_tracks_MP3)]
# filenames_wav = [house_tracks_WAV + f[:-3] + 'wav' for f in listdir(house_tracks_MP3)]

hiphop_tracks_MP3 = '/home/markhyphen/tracks/hiphop/'
hiphop_tracks_WAV = '/home/markhyphen/tracks/hiphop/'
filenames_mp3 = [hiphop_tracks_MP3 + f for f in listdir(hiphop_tracks_MP3)]
filenames_wav = [hiphop_tracks_WAV + f[:-3] + 'wav' for f in listdir(hiphop_tracks_MP3)]
# DnB-junlge DnB_chill DnB_big

# DnB_jungle_tracks_MP3 = '/home/nazgul/temp/Music_MP3/DnB_jungle/'
# DnB_jungle_tracks_WAV = '/home/nazgul/temp/Music_WAV/DnB_jungle/'
# filenames_mp3 = [DnB_jungle_tracks_MP3 + f for f in listdir(DnB_jungle_tracks_MP3)]
# filenames_wav = [DnB_jungle_tracks_WAV + f[:-3] + 'wav' for f in listdir(DnB_jungle_tracks_MP3)]
#
for i, file in enumerate(filenames_mp3):
    print('MP3:', file)
    sound = AudioSegment.from_mp3(file)
    sound.export(filenames_wav[i], format="wav")
    print('WAV:', filenames_wav[i])

# DnB_chill_tracks_MP3 = '/home/nazgul/temp/Music_MP3/DnB_chill/'
# DnB_chill_tracks_WAV = '/home/nazgul/temp/Music_WAV/DnB_chill/'
# filenames_mp3 = [DnB_chill_tracks_MP3 + f for f in listdir(DnB_chill_tracks_MP3)]
# filenames_wav = [DnB_chill_tracks_WAV + f[:-3] + 'wav' for f in listdir(DnB_chill_tracks_MP3)]

# for i, file in enumerate(filenames_mp3):
#     print('MP3:', file)
#     sound = AudioSegment.from_mp3(file)
#     sound.export(filenames_wav[i], format="wav")
#     print('WAV:', filenames_wav[i])
#
# DnB_big_tracks_MP3 = '/home/nazgul/temp/Music_MP3/DnB_big/'
# DnB_big_tracks_WAV = '/home/nazgul/temp/Music_WAV/DnB_big/'
# filenames_mp3 = [DnB_big_tracks_MP3 + f for f in listdir(DnB_big_tracks_MP3)]
# filenames_wav = [DnB_big_tracks_WAV + f[:-3] + 'wav' for f in listdir(DnB_big_tracks_MP3)]
#
# for i, file in enumerate(filenames_mp3):
#     print('MP3:', file)
#     sound = AudioSegment.from_mp3(file)
#     sound.export(filenames_wav[i], format="wav")
#     print('WAV:', filenames_wav[i])
#
# House_progressive_tracks_MP3 = '/home/nazgul/temp/Music_MP3/House_progressive/'
# House_progressive_tracks_WAV = '/home/nazgul/temp/Music_WAV/House_progressive/'
# filenames_mp3 = [House_progressive_tracks_MP3 + f for f in listdir(House_progressive_tracks_MP3)]
# filenames_wav = [House_progressive_tracks_WAV + f[:-3] + 'wav' for f in listdir(House_progressive_tracks_MP3)]
#
# for i, file in enumerate(filenames_mp3):
#     print('MP3:', file)
#     sound = AudioSegment.from_mp3(file)
#     sound.export(filenames_wav[i], format="wav")
#     print('WAV:', filenames_wav[i])
#
# House_acid_tracks_MP3 = '/home/nazgul/temp/Music_MP3/House_acid/'
# House_acid_tracks_WAV = '/home/nazgul/temp/Music_WAV/House_acid/'
# filenames_mp3 = [House_acid_tracks_MP3 + f for f in listdir(House_acid_tracks_MP3)]
# filenames_wav = [House_acid_tracks_WAV + f[:-3] + 'wav' for f in listdir(House_acid_tracks_MP3)]
#
# for i, file in enumerate(filenames_mp3):
#     print('MP3:', file)
#     sound = AudioSegment.from_mp3(file)
#     sound.export(filenames_wav[i], format="wav")
#     print('WAV:', filenames_wav[i])
#
# House_minimal_tracks_MP3 = '/home/nazgul/temp/Music_MP3/House_minimal/'
# House_minimal_tracks_WAV = '/home/nazgul/temp/Music_WAV/House_minimal/'
# filenames_mp3 = [House_minimal_tracks_MP3 + f for f in listdir(House_minimal_tracks_MP3)]
# filenames_wav = [House_minimal_tracks_WAV + f[:-3] + 'wav' for f in listdir(House_minimal_tracks_MP3)]
#
# for i, file in enumerate(filenames_mp3):
#     print('MP3:', file)
#     sound = AudioSegment.from_mp3(file)
#     sound.export(filenames_wav[i], format="wav")
#     print('WAV:', filenames_wav[i])


temp_insert = 0

