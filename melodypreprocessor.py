import json
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer

class MelodyPreprocessor:

    def __init__(self, dataset_path, batch_size=32):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.tokenizer = Tokenizer(filters="", lower=False, split=",")
        self.max_melody_length = None
        self.number_of_tokens = None

    @property
    def number_of_tokens_with_padding(self):
        return self.number_of_tokens + 1

    def create_training_dataset(self):
        dataset = self._load_dataset()
        parsed_melodies = [self._parse_melody(melody) for melody in dataset]
        tokenized_melodies = self._tokenize_and_encode_melodies(parsed_melodies)
        self._set_max_melody_length(tokenized_melodies)
        self._set_number_of_tokens()
        tf_training_dataset = self._convert_to_tf_dataset(tokenized_melodies)
        return tf_training_dataset

    def _load_dataset(self):
        with open(self.dataset_path, "r") as f:
            return json.load(f)

    def _parse_melody(self, melody_str):
        return melody_str.split(", ")

    def _tokenize_and_encode_melodies(self, melodies):
        self.tokenizer.fit_on_texts(melodies)
        tokenized_melodies = self.tokenizer.texts_to_sequences(melodies)
        return tokenized_melodies

    def _set_max_melody_length(self, melodies):
         self.max_melody_length = max(len(melody) for melody in melodies)

    def _set_number_of_tokens(self):
        self.number_of_tokens = len(self.tokenizer.word_index)

    def _sequence_pair_generator(self, melodies):
        for melody in melodies:
            for i in range(1, len(melody)):
                input_seq = melody[:i]
                target_seq = melody[1 : i + 1]
                yield input_seq, target_seq

    def _convert_to_tf_dataset(self, melodies):
        output_signature = (
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
        dataset = tf.data.Dataset.from_generator(
            lambda: self._sequence_pair_generator(melodies),
            output_signature=output_signature,
        )
        dataset = dataset.shuffle(1000)
        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=([self.max_melody_length], [self.max_melody_length]),
            padding_values=(0, 0),
        )
        return dataset

    def fit_tokenizer(self):
        dataset = self._load_dataset()
        parsed_melodies = [self._parse_melody(melody) for melody in dataset]
        self.tokenizer.fit_on_texts(parsed_melodies)
        self.number_of_tokens = len(self.tokenizer.word_index)

def tokens_to_midi(tokens, output_path="output.mid"):
        """
        Converts a list of tokens (e.g., ['C4-1.0', 'E4-1.0']) to a MIDI file.
        """
        from music21 import stream, note
    
        midi_stream = stream.Stream()
        for token in tokens:
            try:
                pitch, duration = token.split('-')
                n = note.Note(pitch)
                n.quarterLength = float(duration)
                midi_stream.append(n)
            except Exception as e:
                print(f"Skipping token {token}: {e}")
    
        midi_stream.write('midi', fp=output_path)
        print(f"MIDI file saved as {output_path}")    

if __name__ == "__main__":
    melody_preprocessor = MelodyPreprocessor("dataset.json", batch_size=32)
    melody_preprocessor.fit_tokenizer()
    vocab_size = melody_preprocessor.number_of_tokens_with_padding
    training_dataset = melody_preprocessor.create_training_dataset()
