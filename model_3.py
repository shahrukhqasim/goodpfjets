import tensorflow as tf


class SimplerGraphLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SimplerGraphLayer, self).__init__()
        self.dense_input_vertices_tracks_transform_1 = tf.keras.layers.Dense(80, activation=tf.nn.relu)
        self.dense_input_vertices_tracks_transform_2 = tf.keras.layers.Dense(80, activation=tf.nn.relu)

        self.dense_input_vertices_ecal_transform_1 = tf.keras.layers.Dense(80, activation=tf.nn.relu)
        self.dense_input_vertices_ecal_transform_2 = tf.keras.layers.Dense(80, activation=tf.nn.relu)

        self.dense_input_vertices_hcal_transform_1 = tf.keras.layers.Dense(80, activation=tf.nn.relu)
        self.dense_input_vertices_hcal_transform_2 = tf.keras.layers.Dense(80, activation=tf.nn.relu)

        self.dense_input_central_transform_1 = tf.keras.layers.Dense(80, activation=tf.nn.relu)
        self.dense_input_central_transform_2 = tf.keras.layers.Dense(80, activation=tf.nn.relu)

    def call(self, vertices_tracks, vertices_ecal, vertices_hcal, central, num_vertices_tracks, num_vertices_ecal, num_vertices_hcal):

        vertices_tracks_x = self.dense_input_vertices_tracks_transform_1(vertices_tracks)
        vertices_tracks_x = self.dense_input_vertices_tracks_transform_2(vertices_tracks_x)

        vertices_ecal_x = self.dense_input_vertices_ecal_transform_1(vertices_ecal)
        vertices_ecal_x = self.dense_input_vertices_ecal_transform_2(vertices_ecal_x)

        vertices_hcal_x = self.dense_input_vertices_hcal_transform_1(vertices_hcal)
        vertices_hcal_x = self.dense_input_vertices_hcal_transform_2(vertices_hcal_x)


        nzero_num_vertices_tracks = tf.where(tf.math.equal(num_vertices_tracks[..., tf.newaxis],0.),1.,num_vertices_tracks[..., tf.newaxis])
        nzero_num_vertices_ecal = tf.where(tf.math.equal(num_vertices_ecal[..., tf.newaxis],0.),1.,num_vertices_ecal[..., tf.newaxis])
        nzero_num_vertices_hcal = tf.where(tf.math.equal(num_vertices_hcal[..., tf.newaxis],0.),1.,num_vertices_hcal[..., tf.newaxis])



        reduced_vertices_tracks_x = tf.reduce_sum(vertices_tracks_x, axis=1) / nzero_num_vertices_tracks
        reduced_vertices_tracks_x = reduced_vertices_tracks_x * tf.cast(tf.math.greater(num_vertices_tracks, 0.)[..., tf.newaxis], tf.float32)
        # reduced_vertices_tracks_x = tf.where(tf.math.is_nan(reduced_vertices_tracks_x), 0, reduced_vertices_tracks_x)
        # reduced_vertices_tracks_x = tf.where(tf.math.is_inf(reduced_vertices_tracks_x), 0, reduced_vertices_tracks_x)


        reduced_vertices_ecal_x = tf.reduce_sum(vertices_ecal_x, axis=1) / nzero_num_vertices_ecal
        reduced_vertices_ecal_x = reduced_vertices_ecal_x * tf.cast(tf.math.greater(num_vertices_ecal, 0.)[..., tf.newaxis], tf.float32)

        # reduced_vertices_ecal_x = tf.where(tf.math.is_nan(reduced_vertices_ecal_x), 0, reduced_vertices_ecal_x)
        # reduced_vertices_ecal_x = tf.where(tf.math.is_inf(reduced_vertices_ecal_x), 0, reduced_vertices_ecal_x)

        reduced_vertices_hcal_x = tf.reduce_sum(vertices_hcal_x, axis=1) / nzero_num_vertices_hcal
        reduced_vertices_hcal_x = reduced_vertices_hcal_x * tf.cast(tf.math.greater(num_vertices_hcal, 0.)[..., tf.newaxis], tf.float32)

        # reduced_vertices_hcal_x = tf.where(tf.math.is_nan(reduced_vertices_hcal_x), 0, reduced_vertices_hcal_x)
        # reduced_vertices_hcal_x = tf.where(tf.math.is_inf(reduced_vertices_hcal_x), 0, reduced_vertices_hcal_x)


        # print(central.shape)
        # print(reduced_vertices_ecal_x.shape)

        central_x = tf.concat((central, reduced_vertices_ecal_x, reduced_vertices_hcal_x, reduced_vertices_tracks_x), axis=-1)
        central_x = self.dense_input_central_transform_1(central_x)
        central_x = self.dense_input_central_transform_2(central_x)


        tiled_central_tracks_x = tf.tile(tf.expand_dims(central_x, axis=1), multiples=[1, vertices_tracks_x.shape[1], 1])
        tiled_central_tracks_x = tiled_central_tracks_x * tf.cast(tf.sequence_mask(num_vertices_tracks, maxlen=vertices_tracks_x.shape[1])[..., tf.newaxis], tf.float32)

        tiled_central_ecal_x = tf.tile(tf.expand_dims(central_x, axis=1), multiples=[1, vertices_ecal_x.shape[1], 1])
        tiled_central_ecal_x = tiled_central_ecal_x * tf.cast(tf.sequence_mask(num_vertices_ecal, maxlen=vertices_ecal_x.shape[1])[..., tf.newaxis], tf.float32)

        tiled_central_hcal_x = tf.tile(tf.expand_dims(central_x, axis=1), multiples=[1, vertices_hcal_x.shape[1], 1])
        tiled_central_hcal_x = tiled_central_hcal_x * tf.cast(tf.sequence_mask(num_vertices_tracks, maxlen=vertices_hcal_x.shape[1])[..., tf.newaxis], tf.float32)


        vertices_tracks_x = tf.concat((tiled_central_tracks_x, vertices_tracks_x), axis=-1)
        vertices_ecal_x = tf.concat((tiled_central_ecal_x, vertices_ecal_x), axis=-1)
        vertices_hcal_x = tf.concat((tiled_central_hcal_x, vertices_hcal_x), axis=-1)

        return vertices_tracks_x, vertices_ecal_x, vertices_hcal_x, central_x


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


    def call(self, vertices_tracks, vertices_ecal, vertices_hcal, central, num_vertices_tracks, num_vertices_ecal, num_vertices_hcal):

        vertices_tracks_x, vertices_ecal_x, vertices_hcal_x, central_x = self.glayer_1(vertices_tracks,vertices_ecal, vertices_hcal, central, num_vertices_tracks, num_vertices_ecal, num_vertices_hcal)
        vertices_tracks_x, vertices_ecal_x, vertices_hcal_x, central_x = self.glayer_2(vertices_tracks_x, vertices_ecal_x, vertices_hcal_x, central_x, num_vertices_tracks, num_vertices_ecal, num_vertices_hcal)
        vertices_tracks_x, vertices_ecal_x, vertices_hcal_x, central_x = self.glayer_3(vertices_tracks_x, vertices_ecal_x, vertices_hcal_x, central_x, num_vertices_tracks, num_vertices_ecal, num_vertices_hcal)
        vertices_tracks_x, vertices_ecal_x, vertices_hcal_x, central_x = self.glayer_4(vertices_tracks_x, vertices_ecal_x, vertices_hcal_x, central_x, num_vertices_tracks, num_vertices_ecal, num_vertices_hcal)
        vertices_tracks_x, vertices_ecal_x, vertices_hcal_x, central_x = self.glayer_5(vertices_tracks_x, vertices_ecal_x, vertices_hcal_x, central_x, num_vertices_tracks, num_vertices_ecal, num_vertices_hcal)


        x = self.output_dense_1(central_x)
        x = self.output_dense_2(x)
        x = self.output_dense_3(x)

        return x



