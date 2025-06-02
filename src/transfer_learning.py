import tensorflow as tf

# Set dataset input dimension (D1: 462, D2: 573)
INPUT_DIM = 462  # or 573 for Dataset D2

# Define the same autoencoder architecture
def create_keras_model(input_dim):
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(80, activation='relu'),
        tf.keras.layers.Dense(60, activation='relu'),
        tf.keras.layers.Dense(40, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu', name="latent"),
        tf.keras.layers.Dense(40, activation='relu'),
        tf.keras.layers.Dense(60, activation='relu'),
        tf.keras.layers.Dense(80, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(input_dim, activation='sigmoid')
    ])

# Simulated global model (assume it is pre-trained)
global_model = create_keras_model(INPUT_DIM)

# Simulated loading of pre-trained weights
# global_model.load_weights('path_to_global_model_weights.h5')

# Freeze encoder layers
for layer in global_model.layers[:5]:  # assuming encoder is first 5 layers
    layer.trainable = False

# Reinitialize decoder layers
for layer in global_model.layers[5:]:
    for w in layer.weights:
        w.assign(tf.random.normal(w.shape))

# Compile model for fine-tuning
global_model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-6),
    loss=tf.keras.losses.MeanSquaredError()
)

# Generate synthetic local dataset (replace with real unseen node data)
x_unseen = tf.random.uniform((100, INPUT_DIM))
local_dataset = tf.data.Dataset.from_tensor_slices((x_unseen, x_unseen)).batch(10)

# Fine-tune model
global_model.fit(local_dataset, epochs=10)

# Save fine-tuned model
# global_model.save_weights('fine_tuned_model_unseen_node.h5')
