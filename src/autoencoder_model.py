import tensorflow as tf

class DenseAutoencoder(tf.keras.Model):
    def __init__(self, input_dim):
        super(DenseAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dense(60, activation='relu'),
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.Dense(20, activation='relu', name="latent")
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.Dense(60, activation='relu'),
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='sigmoid')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
