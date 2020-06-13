import pydub
import glob
import os

def convert(src_p, target_p):
    music_list = sorted(glob.glob(src_p))
    for i, v in enumerate(music_list):
        try:
            mp3 = pydub.AudioSegment.from_mp3(v)
            path = os.path.join(target_p, v.replace(".mp3", ".wav"))
            mp3.export(path, format="wav")
        except pydub.exceptions.CouldntDecodeError:
            print("could not convert file {} to wav".format(v))

src = "/Users/niopeng/Documents/Study/CSE/Spring2020/CSE291E/Fake-Rapper-Helper/Data/mp3/*.mp3"
target = "/Users/niopeng/Documents/Study/CSE/Spring2020/CSE291E/Fake-Rapper-Helper/Data/wav/"
convert(src, target)
