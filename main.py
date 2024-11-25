import os
from pydub import AudioSegment
from spleeter.separator import Separator
import numpy as np
import librosa

def m4a_to_wav(m4a_file, wav_file):
    audio = AudioSegment.from_file(m4a_file, format="mp3")
    audio.export(wav_file, format="wav")

def get_frequencies(wav_file):
    y, sr = librosa.load(wav_file)
    D = np.abs(librosa.stft(y))
    frequencies = librosa.core.fft_frequencies(sr=sr, n_fft=D.shape[0])
    return frequencies, D

def frequency_to_note(freq):
    if freq <= 0:
        return None
    
    A4 = 440
    C0 = A4 * pow(2, -4.75)
    name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    h = round(12 * np.log2(freq / C0))
    octave = h // 12
    n = h % 12
    return name[n] + str(octave)

def separate_audio(m4a_file, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the Spleeter separator with 2 stems (vocals and accompaniment)
    separator = Separator('spleeter:2stems')

    # Perform the separation
    separator.separate_to_file(m4a_file, output_dir)


def analyze_audio(m4a_file):
    wav_file = "temp.wav"
    m4a_to_wav(m4a_file, wav_file)
    frequencies, D = get_frequencies(wav_file)
    
    notes = []
    for i in range(D.shape[1]):
        index = np.argmax(D[:, i])
        freq = frequencies[index]
        note = frequency_to_note(freq)
        if note:
            notes.append(note)
    
    return notes

if __name__ == "__main__":
    m4a_file = "test.mp3"
    # notes = analyze_audio(m4a_file)
    # print(notes)
    output_dir = "output"
    separate_audio(m4a_file, output_dir)
