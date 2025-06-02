import collections
import tensorflow as tf
import tensorflow_federated as tff

# Select dataset input dimension (D1: 462, D2: 573)
INPUT_DIM = 462  # Change to 573 for Dataset D2

# Autoencoder architecture
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

# Create TFF model
def model_fn():
    keras_model = create_keras_model(INPUT_DIM)
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=collections.OrderedDict(
            x=tf.TensorSpec(shape=(None, INPUT_DIM), dtype=tf.float32),
            y=tf.TensorSpec(shape=(None, INPUT_DIM), dtype=tf.float32),
        ),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()]
    )

#  client data for federated training
def generate_client_data(num_clients=5, input_dim=462, num_examples=100):
    client_data = []
    for _ in range(num_clients):
        x = tf.random.uniform((num_examples, input_dim))
        client_data.append({'x': x, 'y': x})
    return [tf.data.Dataset.from_tensor_slices(cd).batch(10) for cd in client_data]

# Fine-tuning transferred model for FTL
def fine_tune_transferred_model(global_model, local_data, input_dim):
    for layer in global_model.layers:
        if "latent" in layer.name or "dense" in layer.name:
            layer.trainable = False  # Freeze encoder
    # Reinitialize decoder
    for layer in global_model.layers[-5:]:
        layer.set_weights([tf.random.normal(w.shape) for w in layer.get_weights()])

    global_model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-6),
        loss=tf.keras.losses.MeanSquaredError()
    )
    global_model.fit(local_data, epochs=10, batch_size=10)
    return global_model

# Federated training using FedAvg
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.RMSprop(learning_rate=1e-6),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
)

# Run federated learning
NUM_ROUNDS = 100
client_datasets = generate_fake_client_data(num_clients=5, input_dim=INPUT_DIM)
state = iterative_process.initialize()
for round_num in range(1, NUM_ROUNDS + 1):
    state, metrics = iterative_process.next(state, client_datasets)
    print(f'Round {round_num}, Metrics={metrics}')

# Extract trained global model
final_model_weights = iterative_process.get_model_weights(state)
global_keras_model = create_keras_model(INPUT_DIM)
final_model_weights.assign_weights_to(global_keras_model)

# Fine-tuning (FTL-style) on unseen client data
print("\n--- Fine-Tuning Global Model on Unseen Node (FTL) ---")
ftl_data = generate_fake_client_data(num_clients=1, input_dim=INPUT_DIM)[0]  # Simulated unseen node
fine_tuned_model = fine_tune_transferred_model(global_keras_model, ftl_data, INPUT_DIM)
