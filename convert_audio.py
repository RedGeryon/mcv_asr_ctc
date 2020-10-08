import pydub
import csv


def convert_audio(fn, sample_rate=16000):
    print(f'Converting {fn} to {sample_rate} wav')
    # Convert all mp3s into wave files with 16k sample rate
    fp = './en/clips/' + fn
    audio = pydub.AudioSegment.from_mp3(fp)
    audio = audio.set_frame_rate(sample_rate)
    dst = './en/converted/' + fn[:-3] + "wav"
    audio.export(dst, format="wav")


def convert_csv(fp):
    with open(fp) as csvfile:
        reader = csv.reader(csvfile)
        for fp, label in reader:
            convert_audio(fp)


if __name__ == '__main__':
    train_data = './en/train_clean.csv'
    val_data = './en/val_clean.csv'
    test_data = './en/test_clean.csv'
    convert_csv(train_data)
    convert_csv(val_data)
    convert_csv(test_data)
