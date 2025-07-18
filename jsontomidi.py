import json
from mido import Message, MidiFile, MidiTrack
from music21 import note

def json_to_midi(json_path, midi_path):
    with open(json_path, 'r') as f:
        melody_str = json.load(f)  # This is a single string
        tokens = [t.strip().strip('"') for t in melody_str.split(",")]

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    velocity = 64
    time = 240  # 240 ticks per note

    for token in tokens:
        if '-' in token:
            pitch, duration = token.split('-')
            try:
                midi_num = note.Note(pitch).pitch.midi
            except Exception:
                continue
            track.append(Message('note_on', note=midi_num, velocity=velocity, time=0))
            track.append(Message('note_off', note=midi_num, velocity=0, time=int(float(duration) * time)))
    
    mid.save(midi_path)
    print(f"MIDI saved to {midi_path}")

if __name__ == "__main__":
    json_to_midi('generated_melody.json', 'output.mid')