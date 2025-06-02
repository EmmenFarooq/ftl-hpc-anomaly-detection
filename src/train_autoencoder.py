
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from autoencoder_model import DenseAutoencoder, AutoencoderTrainer

def load_dataset(csv_path, test_size=0.2):
    data = pd.read_csv(csv_path)
    data = data.dropna()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return train_test_split(data_scaled, test_size=test_size, random_state=42)

def detect_anomalies(reconstruction_errors, threshold=0.5):
    return reconstruction_errors > threshold

def compute_reconstruction_errors(model, x_data):
    reconstructed = model(x_data, training=False)
    errors = tf.reduce_mean(tf.math.squared_difference(x_data, reconstructed), axis=1)
    return errors.numpy()

def main():
    # === Config ===
    csv_path = 'data/sample_node_data.csv'  # Replace with your actual file
    input_dim = 462                        # Or 573 depending on D1 or D2
    batch_size = 32
    epochs = 10
    threshold = 0.5
    use_dropout = True
    use_batchnorm = False

    # === Load and prepare data ===
    x_train, x_test = load_dataset(csv_path)
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)

    # === Initialize model and trainer ===
    autoencoder = DenseAutoencoder(input_dim, use_batchnorm=use_batchnorm, use_dropout=use_dropout)
    trainer = AutoencoderTrainer(autoencoder)

    # === Training Loop ===
    for epoch in range(epochs):
        trainer.train_loss.reset_state()
        for x_batch in train_dataset:
            trainer.train_step(x_batch.astype("float32"))
        print(f"Epoch {epoch+1}, Loss: {trainer.train_loss.result().numpy():.6f}")

    # === Evaluation and Anomaly Detection ===
    x_test_tf = tf.convert_to_tensor(x_test.astype("float32"))
    errors = compute_reconstruction_errors(autoencoder, x_test_tf)
    anomalies = detect_anomalies(errors, threshold=threshold)

    print(f"Detected {np.sum(anomalies)} anomalies out of {len(errors)} samples.")

if __name__ == '__main__':
    main()
