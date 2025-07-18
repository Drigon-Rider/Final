import tensorflow as tf


class MelodyGenerator:
  
    def __init__(self, transformer, tokenizer, max_length=50):
 
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.max_length = max_length

    def generate(self, start_sequence):
 
        input_tensor = self._get_input_tensor(start_sequence)

        num_notes_to_generate = self.max_length - len(input_tensor[0])

        for _ in range(num_notes_to_generate):
            predictions = self.transformer(
                input_tensor, input_tensor, False, None, None, None
            )
            predicted_note = self._get_note_with_highest_score(predictions)
            input_tensor = self._append_predicted_note(
                input_tensor, predicted_note
            )

        generated_melody = self._decode_generated_sequence(input_tensor)

        return generated_melody

    def _get_input_tensor(self, start_sequence):
   
        input_sequence = self.tokenizer.texts_to_sequences([start_sequence])
        input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.int64)
        return input_tensor

    def _get_note_with_highest_score(self, predictions):

        latest_predictions = predictions[:, -1, :]
        predicted_note_index = tf.argmax(latest_predictions, axis=1)
        predicted_note = predicted_note_index.numpy()[0]
        return predicted_note

    def _append_predicted_note(self, input_tensor, predicted_note):
       
        return tf.concat([input_tensor, [[predicted_note]]], axis=-1)

    def _decode_generated_sequence(self, generated_sequence):
       
        generated_sequence_array = generated_sequence.numpy()
        generated_melody = self.tokenizer.sequences_to_texts(
            generated_sequence_array
        )[0]
        return generated_melody
