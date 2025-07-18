import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from melodygenerator import MelodyGenerator
from melodypreprocessor import MelodyPreprocessor
from transformer import Transformer

# Configure GPU memory growth to prevent OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

EPOCHS = 10
BATCH_SIZE = 8  # Reduced from 32 to 8 to fit in GPU memory
DATA_PATH = "dataset.json"
MAX_SEQUENCE_LENGTH = 1024  # Reduced from 2300 to 1024 for memory efficiency
MAX_POSITIONS_IN_POSITIONAL_ENCODING = 1024  # Match the sequence length

sparse_categorical_crossentropy = SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)
optimizer = Adam()

def train(train_dataset, transformer, epochs):
    print("Training the model...")
    for epoch in range(epochs):
        total_loss = 0
        for (batch, (input, target)) in enumerate(train_dataset):
            batch_loss = _train_step(input, target, transformer)
            total_loss += batch_loss
            print(
                f"Epoch {epoch + 1} Batch {batch + 1} Loss {batch_loss.numpy()}"
            )

@tf.function
def _train_step(input, target, transformer):
    target_input = _right_pad_sequence_once(target[:, :-1])
    target_real = _right_pad_sequence_once(target[:, 1:])
    with tf.GradientTape() as tape:
        predictions = transformer(input, target_input, True, None, None, None)
        loss = _calculate_loss(target_real, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    gradient_variable_pairs = zip(gradients, transformer.trainable_variables)
    optimizer.apply_gradients(gradient_variable_pairs)
    return loss

def _calculate_loss(real, pred):
    loss_ = sparse_categorical_crossentropy(real, pred)
    boolean_mask = tf.math.equal(real, 0)
    mask = tf.math.logical_not(boolean_mask)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    total_loss = tf.reduce_sum(loss_)
    number_of_non_padded_elements = tf.reduce_sum(mask)
    average_loss = total_loss / number_of_non_padded_elements
    return average_loss

def _right_pad_sequence_once(sequence):
    return tf.pad(sequence, [[0, 0], [0, 1]], "CONSTANT")

def save_model(transformer, optimizer, checkpoint_path="transformer_ckpt"):
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt.save(checkpoint_path)
    print(f"Model saved at {checkpoint_path}")

def load_model(transformer, optimizer, checkpoint_path="transformer_ckpt"):
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    latest_ckpt = tf.train.latest_checkpoint('.')
    if latest_ckpt:
        ckpt.restore(latest_ckpt).expect_partial()
        print(f"Model loaded from {latest_ckpt}")
    else:
        print("No checkpoint found.")

if __name__ == "__main__":
    melody_preprocessor = MelodyPreprocessor(DATA_PATH, batch_size=BATCH_SIZE)
    
    # First load to get vocab, but limit sequences to our GPU memory capacity
    dataset = melody_preprocessor._load_dataset()
    parsed_melodies = [melody_preprocessor._parse_melody(melody) for melody in dataset]
    
    # Truncate long sequences to fit in GPU memory
    truncated_melodies = []
    for melody in parsed_melodies:
        if len(melody) > MAX_SEQUENCE_LENGTH:
            melody = melody[:MAX_SEQUENCE_LENGTH]
        truncated_melodies.append(melody)
    
    # Tokenize the truncated melodies
    tokenized_melodies = melody_preprocessor._tokenize_and_encode_melodies(truncated_melodies)
    melody_preprocessor._set_max_melody_length(tokenized_melodies)
    melody_preprocessor._set_number_of_tokens()
    
    print(f"Using max sequence length: {melody_preprocessor.max_melody_length}")
    
    # Create the training dataset
    train_dataset = melody_preprocessor._convert_to_tf_dataset(tokenized_melodies)
    
    vocab_size = melody_preprocessor.number_of_tokens_with_padding
    transformer_model = Transformer(
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_feedforward=128,
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        max_num_positions_in_pe_encoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        max_num_positions_in_pe_decoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        dropout_rate=0.1,
    )
    train(train_dataset, transformer_model, EPOCHS)
    transformer_model.save_weights("transformer_weights.h5")
    print("Model weights saved.")
    melody_generator = MelodyGenerator(
        transformer_model, melody_preprocessor.tokenizer
    )
    # start_sequence = ["C4-1.0", "D4-1.0", "E4-1.0", "C4-1.0"]
    # new_melody = melody_generator.generate(start_sequence)
    # print(f"Generated melody: {new_melody}")