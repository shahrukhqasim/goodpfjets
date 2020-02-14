import tensorflow as tf



def gauss(x):
    return tf.exp(-1* x*x)

class GlobalExchange(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalExchange, self).__init__(**kwargs)

    def build(self, input_shape):
        # tf.ragged FIXME?
        self.num_vertices = input_shape[1]
        super(GlobalExchange, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        # tf.ragged FIXME?
        # maybe just use tf.shape(x)[1] instead?
        mean = tf.tile(mean, [1, self.num_vertices, 1])
        return tf.concat([x, mean], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (input_shape[2] * 2,)

class GarNet2(tf.keras.layers.Layer):
    def __init__(self, n_aggregators=4, n_prop=48, n_mid=48, **kwargs):
        super(GarNet2, self).__init__(**kwargs)
        self.n_aggregators=n_aggregators
        self.n_mid = n_mid
        self.n_prop = n_prop

        self.trans1 = []
        self.trans2 = []
        self.trans3 = []
        self.trans4 = []

        for i in range(self.n_aggregators):
            self.trans1.append(tf.keras.layers.Dense(n_mid, activation=tf.nn.leaky_relu))
            self.trans2.append(tf.keras.layers.Dense(n_mid, activation=tf.nn.leaky_relu))
            self.trans3.append(tf.keras.layers.Dense(n_mid, activation=tf.nn.leaky_relu))
            self.trans4.append(tf.keras.layers.Dense(n_prop, activation=tf.nn.leaky_relu))

    def build(self, input_shape):
        super(GarNet, self).build(input_shape)


        mid_shape = input_shape.copy()
        mid_shape[1] = self.n_mid

        for i in range(self.n_aggregators):
            self.trans1[i].build(input_shape=input_shape)
            self.trans2[i].build(input_shape=mid_shape)
            self.trans3[i].build(input_shape=mid_shape)
            self.trans4[i].build(input_shape=mid_shape)



    def call(self, x):
        for i in range(self.n_aggregators):
            x = self.trans1[i](x)
            x = self.trans2[i](x)

            y = tf.reduce_sum(x, axis=1, keepdims=True)

            x = (x-y)
            x = self.transform3[i](x)
            x = self.transform4[i](x)


        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n_prop)


class GarNet(tf.keras.layers.Layer):
    def __init__(self, n_aggregators, n_filters, n_propagate, **kwargs):
        super(GarNet, self).__init__(**kwargs)

        self.n_aggregators = n_aggregators
        self.n_filters = n_filters
        self.n_propagate = n_propagate

        self.input_feature_transform = tf.keras.layers.Dense(n_propagate)
        self.aggregator_distance = tf.keras.layers.Dense(n_aggregators)
        self.output_feature_transform = tf.keras.layers.Dense(n_filters, activation='relu')

        self._sublayers = [self.input_feature_transform, self.aggregator_distance, self.output_feature_transform]

    def build(self, input_shape):
        self.input_feature_transform.build(input_shape)
        self.aggregator_distance.build(input_shape)

        # tf.ragged FIXME? tf.shape()?
        self.output_feature_transform.build((input_shape[0], input_shape[1], input_shape[
            2] + self.aggregator_distance.units + 2 * self.aggregator_distance.units * (
                                                         self.input_feature_transform.units + self.aggregator_distance.units)))

        for layer in self._sublayers:
            self._trainable_weights.extend(layer.trainable_weights)
            self._non_trainable_weights.extend(layer.non_trainable_weights)

        super(GarNet, self).build(input_shape)

    def call(self, x):
        features = self.input_feature_transform(x)  # (B, V, F)
        distance = self.aggregator_distance(x)  # (B, V, S)

        edge_weights = gauss(distance)

        features = tf.concat([features, edge_weights], axis=-1)  # (B, V, F+S)

        # vertices -> aggregators
        edge_weights_trans = tf.transpose(edge_weights, perm=(0, 2, 1))  # (B, S, V)
        aggregated_max = self.apply_edge_weights(features, edge_weights_trans, aggregation=tf.reduce_max)  # (B, S, F+S)
        aggregated_mean = self.apply_edge_weights(features, edge_weights_trans,
                                                  aggregation=tf.reduce_mean)  # (B, S, F+S)

        aggregated = tf.concat([aggregated_max, aggregated_mean], axis=-1)  # (B, S, 2*(F+S))

        # aggregators -> vertices
        updated_features = self.apply_edge_weights(aggregated, edge_weights)  # (B, V, 2*S*(F+S))

        updated_features = tf.concat([x, updated_features, edge_weights], axis=-1)  # (B, V, X+2*S*(F+S)+S)

        return self.output_feature_transform(updated_features)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_feature_transform.units)

    def apply_edge_weights(self, features, edge_weights, aggregation=None):
        features = tf.expand_dims(features, axis=1)  # (B, 1, v, f)
        edge_weights = tf.expand_dims(edge_weights, axis=3)  # (B, A, v, 1)

        # tf.ragged FIXME? broadcasting should work
        out = edge_weights * features  # (B, u, v, f)
        # tf.ragged FIXME? these values won't work
        n = features.shape[-2] * features.shape[-1]

        if aggregation:
            out = aggregation(out, axis=2)  # (B, u, f)
            n = features.shape[-1]

        # tf.ragged FIXME? there might be a chance to spell out batch dim instead and use -1 for vertices?
        return tf.reshape(out, [-1, out.shape[1], n])  # (B, u, n)

    def get_config(self):
        config = {'n_aggregators': self.n_aggregators, 'n_filters': self.n_filters, 'n_propagate': self.n_propagate,
                  'name': self.name}
        base_config = super(GarNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


#
# class DummyModel(tf.keras.Model):
#     def __init__(self,**kwargs):
#         super(DummyModel, self).__init__(**kwargs)
#
#         self.ourdense =
#
#     def call(self, x, calo_in, target):
#         pass






class GarNetClusteringModel2(tf.keras.Model):
    def __init__(self, **kwargs):
        super(GarNetClusteringModel2, self).__init__(**kwargs)

        self.blocks = []

        momentum = kwargs.get('momentum', 0.99)

        self.input_gex = self.add_layer(GlobalExchange)
        self.input_batchnorm = self.add_layer(tf.keras.layers.BatchNormalization, momentum=momentum)
        self.input_dense = self.add_layer(tf.keras.layers.Dense, 48, activation='relu')

        for i, (n_aggregators, n_filters, n_propagate) in enumerate(block_params):
            garnet = self.add_layer(GarNet, n_aggregators, n_filters, n_propagate)
            batchnorm = self.add_layer(tf.keras.layers.BatchNormalization, momentum=momentum)

            self.blocks.append((garnet,batchnorm))

        self.output_dense_0 = self.add_layer(tf.keras.layers.Dense, 96, activation='relu')
        self.output_dense_01 = self.add_layer(tf.keras.layers.Dense, 96, activation='relu')
        self.output_dense_1 = self.add_layer(tf.keras.layers.Dense, 1, activation=None, use_bias=False)


        self.input_dense_calo_0 = self.add_layer(tf.keras.layers.Dense, 48, activation='relu')
        self.input_dense_calo_1 = self.add_layer(tf.keras.layers.Dense, 48, activation='relu')
        self.input_dense_calo_2 = self.add_layer(tf.keras.layers.Dense, 32, activation=None)

    def call(self, x, calo_in):
        feats = []

        # x, calo_in = inputs


        x = self.input_gex(x)
        x = self.input_dense(x)



        y = self.input_dense_calo_0(calo_in)
        y = self.input_dense_calo_1(y)
        y = self.input_dense_calo_2(y)

        y = tf.expand_dims(y, axis=1)
        y = tf.tile(y, multiples=[1,100,1])


        x = tf.concat((x,y), axis=-1)
        x = self.input_batchnorm(x)



        for block in self.blocks:
            for layer in block:
                x = layer(x)

            feats.append(x)


        x = tf.concat(feats, axis=-1)



        # x = tf.concat((x*0,y), axis=-1)

        # x = self.input_batchnorm(calo_in)
        x = tf.concat((x,y), axis=-1)
        x = self.output_dense_0(x)
        x = self.output_dense_01(x)
        x = self.output_dense_1(x)
        x = tf.reduce_sum(x, axis=1)

        # print(self.output_dense_1.get_weights())


        return x

    def add_layer(self, cls, *args, **kwargs):
        layer = cls(*args, **kwargs)
        self._layers.append(layer)
        return layer





class GarNetClusteringModel(tf.keras.Model):
    def __init__(self, aggregators=([4] * 11), filters=([32] * 11), propagate=([20] * 11), **kwargs):
        super(GarNetClusteringModel, self).__init__(**kwargs)

        self.blocks = []

        block_params = zip(aggregators, filters, propagate)

        momentum = kwargs.get('momentum', 0.99)

        self.input_gex = self.add_layer(GlobalExchange)
        self.input_batchnorm = self.add_layer(tf.keras.layers.BatchNormalization, momentum=momentum)
        self.input_dense = self.add_layer(tf.keras.layers.Dense, 32, activation='relu')

        for i, (n_aggregators, n_filters, n_propagate) in enumerate(block_params):
            garnet = self.add_layer(GarNet, n_aggregators, n_filters, n_propagate)
            batchnorm = self.add_layer(tf.keras.layers.BatchNormalization, momentum=momentum)

            self.blocks.append((garnet,batchnorm))

        self.output_dense_0 = self.add_layer(tf.keras.layers.Dense, 96, activation='relu')
        self.output_dense_01 = self.add_layer(tf.keras.layers.Dense, 96, activation='relu')
        self.output_dense_1 = self.add_layer(tf.keras.layers.Dense, 1, activation=None, use_bias=False)


        self.input_dense_calo_0 = self.add_layer(tf.keras.layers.Dense, 48, activation='relu')
        self.input_dense_calo_1 = self.add_layer(tf.keras.layers.Dense, 48, activation='relu')
        self.input_dense_calo_2 = self.add_layer(tf.keras.layers.Dense, 32, activation=None)

    def call(self, x, calo_in):
        feats = []

        # x, calo_in = inputs


        x = self.input_gex(x)
        x = self.input_dense(x)



        y = self.input_dense_calo_0(calo_in)
        y = self.input_dense_calo_1(y)
        y = self.input_dense_calo_2(y)

        y = tf.expand_dims(y, axis=1)
        y = tf.tile(y, multiples=[1,100,1])


        x = tf.concat((x,y), axis=-1)
        x = self.input_batchnorm(x)



        for block in self.blocks:
            for layer in block:
                x = layer(x)

            feats.append(x)


        x = tf.concat(feats, axis=-1)



        # x = tf.concat((x*0,y), axis=-1)

        # x = self.input_batchnorm(calo_in)
        x = tf.concat((x,y), axis=-1)
        x = self.output_dense_0(x)
        x = self.output_dense_01(x)
        x = self.output_dense_1(x)
        x = tf.reduce_sum(x, axis=1)

        # print(self.output_dense_1.get_weights())


        return x

    def add_layer(self, cls, *args, **kwargs):
        layer = cls(*args, **kwargs)
        self._layers.append(layer)
        return layer




