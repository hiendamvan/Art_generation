import tensorflow as tf
import numpy as np
import cv2

class NeuralStyleTransfer:
    def __init__(self, content_image, style_image):
        self.content_image = tf.Variable(content_image, dtype=tf.float32)
        self.style_image = tf.Variable(style_image, dtype=tf.float32)
        self.generated_image = tf.Variable(content_image, dtype=tf.float32)
        self.model = self.get_feature_extractor()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5.0)
    
    def get_feature_extractor(self):
        content_layer = [('block5_conv4', 1)]
        STYLE_LAYERS = [
            ('block1_conv1', 0.2),
            ('block2_conv1', 0.2),
            ('block3_conv1', 0.2),
            ('block4_conv1', 0.2),
            ('block5_conv1', 0.2)]
        vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(400, 400, 3),
                                  weights='Art_generation/pretrained_model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

        vgg.trainable = False
        outputs = [vgg.get_layer(layer[0]).output for layer in STYLE_LAYERS + content_layer]

        model = tf.keras.Model([vgg.input], outputs)
        return model


    def compute_content_loss(self, content_output, target_output):
        return tf.reduce_mean(tf.square(content_output - target_output))
    
    def gram_matrix(self, feature_maps):
        gram = tf.linalg.einsum("bijc,bijd->bcd", feature_maps, feature_maps)
        return gram / tf.cast(tf.shape(feature_maps)[1] * tf.shape(feature_maps)[2], tf.float32)
    
    def compute_style_loss(self, style_output, target_output):
        return tf.reduce_mean(tf.square(self.gram_matrix(style_output) - self.gram_matrix(target_output)))
    
    def compute_total_loss(self):
        content_target = self.model(self.content_image)[0]
        style_target = self.model(self.style_image)[1:]
        generated_features = self.model(self.generated_image)
        content_loss = self.compute_content_loss(generated_features[0], content_target)
        style_loss = sum([self.compute_style_loss(generated_features[i+1], style_target[i]) for i in range(len(style_target))])
        return content_loss * 1.0 + style_loss * 1e-4
    
    def train_style_transfer(self, num_iterations=10):
        for i in range(num_iterations):
            with tf.GradientTape() as tape:
                loss = self.compute_total_loss()
            grads = tape.gradient(loss, self.generated_image)
            self.optimizer.apply_gradients([(grads, self.generated_image)])
            print(f"Iteration {i}: Loss = {loss.numpy()}")
        return self.deprocess_image(self.generated_image)
    
    def deprocess_image(self, image):
        image = image.numpy()
        image = np.squeeze(image, axis=0)
        image = np.clip(image, 0, 255).astype("uint8")
        return image