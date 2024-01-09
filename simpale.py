import numpy as np
import tensorflow as tf
import os
import soundfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications import ResNet50


def read_audio(path, target_fs):

    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs


def melspec(path, target_fs, savepath):

    filelist = os.listdir(path)
    for elem in filelist:
        filename = os.listdir(path + elem + '/')
        for i in range(0, len(filename)):
            audio, fs = read_audio(path + elem + '/' + filename[i], target_fs)
            melspec = librosa.feature.melspectrogram(y=audio / 32768, sr=fs, n_fft=1024, hop_length=512, n_mels=128,
                                                     power=2)
            logmelspec = librosa.power_to_db(melspec)
            mfcc = librosa.feature.mfcc(y=audio, sr=fs)
            plt.figure()
            librosa.display.specshow(logmelspec, sr=fs, x_axis='time', y_axis='hz')
            plt.set_cmap('rainbow')
            plt.savefig(savepath + elem + str(i) + '.png')
            plt.close()
    return None


def transfer_feature(x):
    m = x.shape[0]
    Resnet = ResNet50(include_top=False)
    feature = Resnet.predict(x)
    x_out = feature.reshape(m, -1)
    return x_out


def predict_snoring(model_path, audio_path, target_fs, mel_save_path):
    files = os.listdir(audio_path)
    wav_file_name = ""
    for file in files:
        if file.endswith(".wav"):
            wav_file_name = file
    if not wav_file_name:
        return
    audio, fs = read_audio(audio_path + wav_file_name, target_fs)

    melspec = librosa.feature.melspectrogram(y=audio / 32768, sr=fs, n_fft=1024, hop_length=512, n_mels=128, power=2)
    logmelspec = librosa.power_to_db(melspec)

    plt.figure()
    librosa.display.specshow(logmelspec, sr=fs, x_axis='time', y_axis='hz')
    plt.set_cmap('rainbow')
    plt.savefig(mel_save_path)
    plt.close()

    img = Image.open(mel_save_path)
    img = img.convert("RGB")
    width = img.size[0]
    height = img.size[1]
    rate = 256 / height
    img = img.resize((int(width * rate), int(height * rate)), Image.ANTIALIAS)
    img = img.crop((59, 32, 283, 256))
    img = np.array(img)
    img[:, :, 0] = img[:, :, 0] - np.mean(img[:, :, 0])
    img[:, :, 1] = img[:, :, 1] - np.mean(img[:, :, 1])
    img[:, :, 2] = img[:, :, 2] - np.mean(img[:, :, 2])
    img = np.expand_dims(img, axis=0)

    model = tf.keras.models.load_model(model_path)

    x_out = transfer_feature(img)
    y_prob = model.predict(x_out)

    return y_prob[0][1]


if __name__ == "__main__":

    num1 = 0.1538
    num2 = 0.1709
    model_path = "model/my_model.h5"
    audio_path = "audio/"

    pre = predict_snoring(model_path, audio_path, 16000, "1.png")
    os.remove("1.png")
    a = b = 0
    if pre >= num1:
        a = 1
    if pre >= num2:
        b = 1
    if a and b:
        print("打鼾")
    elif a or b:
        print("打鼾")
    else:
        print("没有打鼾")

