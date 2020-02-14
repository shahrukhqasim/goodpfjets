import tensorflow as tf


class SimplerGraphLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SimplerGraphLayer, self).__init__()
        self.dense_input_vertices_transform_1 = tf.keras.layers.Dense(80, activation=tf.nn.relu)
        self.dense_input_vertices_transform_2 = tf.keras.layers.Dense(80, activation=tf.nn.relu)

        self.dense_input_central_transform_1 = tf.keras.layers.Dense(80, activation=tf.nn.relu)
        self.dense_input_central_transform_2 = tf.keras.layers.Dense(80, activation=tf.nn.relu)

    def call(self, vertices, central, num_vertices):

        vertices_x = self.dense_input_vertices_transform_1(vertices)
        vertices_x = self.dense_input_vertices_transform_2(vertices_x)

        central_x = self.dense_input_central_transform_1(central)
        central_x = self.dense_input_central_transform_2(central_x)


        difference_x = vertices_x - tf.expand_dims(central_x, axis=1)

        difference_x_contracted = difference_x * tf.cast(tf.sequence_mask(num_vertices, maxlen=vertices.shape[1])[..., tf.newaxis], tf.float32)
        difference_x_contracted = tf.reduce_sum(difference_x_contracted, axis=1) / num_vertices[..., tf.newaxis]
        difference_x_contracted_tiled = tf.tile(tf.expand_dims(difference_x_contracted, axis=1), multiples=[1, vertices.shape[1], 1])


        central_x = tf.concat((central_x, difference_x_contracted), axis=-1)
        vertices_x = tf.concat((vertices_x, difference_x_contracted_tiled), axis=-1)

        return vertices_x, central_x


class SimplerGraphModel(tf.keras.models.Model):
    def __init__(self, num_outputs):
        super(SimplerGraphModel, self).__init__()
        self.glayer_1 = SimplerGraphLayer()
        self.glayer_2 = SimplerGraphLayer()
        self.glayer_3 = SimplerGraphLayer()
        self.glayer_4 = SimplerGraphLayer()
        self.glayer_5 = SimplerGraphLayer()

        self.output_dense_1 = tf.keras.layers.Dense(80, activation=tf.nn.relu)
        self.output_dense_2 = tf.keras.layers.Dense(80, activation=tf.nn.relu)
        self.output_dense_3 = tf.keras.layers.Dense(num_outputs)


    def call(self, vertices, central, num_vertices):
        vertices_x, central_x = self.glayer_1(vertices, central, num_vertices)
        vertices_x, central_x = self.glayer_2(vertices_x, central_x, num_vertices)
        vertices_x, central_x = self.glayer_3(vertices_x, central_x, num_vertices)
        vertices_x, central_x = self.glayer_4(vertices_x, central_x, num_vertices)
        vertices_x, central_x = self.glayer_5(vertices_x, central_x, num_vertices)

        x = self.output_dense_1(central_x)
        x = self.output_dense_2(x)
        x = self.output_dense_3(x)

        return x



