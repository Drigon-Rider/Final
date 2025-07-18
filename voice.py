import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from music21 import note, chord
from collections import Counter
import json

def get_start_sequence(filename="humming.wav", duration=5, fs=44100):
    print("Start humming a chord (5 seconds)...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    print(f"Recording saved as {filename}")

    # --- Step 2: Extract Pitches ---
    audio, sr = librosa.load(filename)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)

    pitch_values = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            pitch_values.append(pitch)

    if not pitch_values:
        print("No pitches detected. Try humming louder or closer to the mic.")
        return []

    midi_notes = [librosa.hz_to_midi(p) for p in pitch_values]
    midi_notes = [int(round(m)) for m in midi_notes if 21 <= m <= 108]

    if not midi_notes:
        print("No valid MIDI notes found.")
        return []

    # Get most common 4 MIDI notes
    note_counts = Counter(midi_notes)
    top_notes = [note for note, _ in note_counts.most_common(4)]

    note_names = [note.Note(m).nameWithOctave for m in top_notes]

    if note_names:
        start_sequence = [f"{n}-1.0" for n in note_names]
        with open("start_sequence.json", "w") as outfile:
            json.dump(start_sequence, outfile)
        print('start_sequence = ' + ', '.join([f'"{n}"' for n in start_sequence]))

        if len(note_names) >= 2:
            detected_chord = chord.Chord(note_names)
            print("Detected chord:", detected_chord.commonName)
        else:
            print("Not enough notes detected to form a chord.")

        return start_sequence
    else:
        print("No notes detected.")
        return []

if __name__ == "__main__":
    start_sequence = get_start_sequence()
