import tensorflow as tf


def generate_img():
    latent_dim = 200
    model = tf.saved_model.load('./mixed_model')
    latent_sample = tf.random.normal(shape=(1, latent_dim))
    output = model.decoder(latent_sample)
    return output.numpy()
