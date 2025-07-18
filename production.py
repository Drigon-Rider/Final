import tensorflow as tf
from melodygenerator import MelodyGenerator
from melodypreprocessor import MelodyPreprocessor
from transformer import Transformer
import numpy as np
from voice import get_start_sequence
from melodypreprocessor import tokens_to_midi
import json

BATCH_SIZE = 32
DATA_PATH = "dataset.json"
MAX_SEQUENCE_LENGTH = 1024  # Match training script
MAX_POSITIONS_IN_POSITIONAL_ENCODING = 1024  # Match training script

if __name__ == "__main__":
    melody_preprocessor = MelodyPreprocessor(DATA_PATH, batch_size=BATCH_SIZE)
    melody_preprocessor.fit_tokenizer()
    
    # Process data the same way as training script
    dataset = melody_preprocessor._load_dataset()
    parsed_melodies = [melody_preprocessor._parse_melody(melody) for melody in dataset]
    
    # Truncate long sequences to match training
    truncated_melodies = []
    for melody in parsed_melodies:
        if len(melody) > MAX_SEQUENCE_LENGTH:
            melody = melody[:MAX_SEQUENCE_LENGTH]
        truncated_melodies.append(melody)
    
    # Tokenize the truncated melodies
    tokenized_melodies = melody_preprocessor._tokenize_and_encode_melodies(truncated_melodies)
    melody_preprocessor._set_max_melody_length(tokenized_melodies)
    melody_preprocessor._set_number_of_tokens()
    
    vocab_size = melody_preprocessor.number_of_tokens_with_padding
    # Use the vocabulary size that matches the saved weights (4384)
    # rather than the current tokenizer size (4613)
    vocab_size_for_model = 4384  # This matches the saved weights
    print(f"Production vocab size: {vocab_size}")
    print(f"Using vocab size for model: {vocab_size_for_model}")

    transformer_model = Transformer(
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_feedforward=128,
        input_vocab_size=vocab_size_for_model,
        target_vocab_size=vocab_size_for_model,
        max_num_positions_in_pe_encoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        max_num_positions_in_pe_decoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        dropout_rate=0.1,
    )

    # Build model by calling it on dummy input with all required arguments
    dummy_input = np.zeros((1, 10), dtype=np.int32)
    dummy_target = np.zeros((1, 10), dtype=np.int32)
    dummy_mask = np.ones((1, 1, 1, 10), dtype=np.float32)

    transformer_model(
        input=dummy_input,
        target=dummy_target,
        enc_padding_mask=dummy_mask,
        look_ahead_mask=dummy_mask,
        dec_padding_mask=dummy_mask,
        training=False
    )

    transformer_model.load_weights("transformer_weights.h5")
    print("Model weights loaded.")

    melody_generator = MelodyGenerator(
        transformer_model, melody_preprocessor.tokenizer
    )
    
    with open("start_sequence.json", "r")as f:
        start_sequence = json.load(f)
    
    print("Loaded start_sequence:", start_sequence)
    new_melody = melody_generator.generate(start_sequence)
    print(f"Generated melody: {new_melody}")
    new_melody = ', '.join(new_melody.strip().split())
    with open("generated_melody.json", "w") as outfile:
        json.dump(new_melody, outfile)
    print("Generated melody saved to generated_melody.json")