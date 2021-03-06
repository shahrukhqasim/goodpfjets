import tensorflow as tf
import helpers_tf as htf
import numpy as np
from model_2 import SimplerGraphModel

batch_size = 1024

train_set = tf.data.TFRecordDataset(['train.tfrecords'], "GZIP")
train_set = train_set.map(htf.extract_fn)
train_set = train_set.shuffle(buffer_size=5000)
train_set = train_set.batch(batch_size)




# next_element = iterator.get_next()
val_set = tf.data.TFRecordDataset(['val.tfrecords'], "GZIP")
test_set = tf.data.TFRecordDataset(['test.tfrecords'], "GZIP")



the_model = SimplerGraphModel(num_outputs=1)
the_model.call(np.zeros((batch_size, 100, 6)), np.zeros((batch_size, 7)), np.ones((batch_size,)))



if False:
    the_model.load_weights('checkpoints/land_cruiser')

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)



writer = tf.summary.create_file_writer("summaries_def/the_one")



mean_vector = np.array([59.384735, 0.0053087026, -0.0077821156, -0.5366358])[np.newaxis, ...]
mean_vector_t = tf.cast(tf.constant(mean_vector), tf.float32)

scaling_vector = np.array([210.64401, 10.853447, 11.070456, 218.28716])[np.newaxis, ...]
scaling_vector_t = tf.cast(tf.constant(scaling_vector), tf.float32)

#
# xyz = None
# for next_element in train_set:
#     xyz = next_element
#     break


with writer.as_default():
    iteration = 0
    for epoch in range(500):
        print("Starting epoch", epoch)
        for next_element in train_set:
            if len(next_element['tracks_px']) < batch_size:
                continue

            tracks_in = tf.concat((next_element['tracks_px'][..., tf.newaxis],
                                       next_element['tracks_py'][..., tf.newaxis],
                                       next_element['tracks_pz'][..., tf.newaxis],
                                       next_element['tracks_eta'][..., tf.newaxis],
                                       next_element['tracks_phi'][..., tf.newaxis],
                                       next_element['tracks_pt'][..., tf.newaxis],), axis=-1)

            calo_in =  tf.concat((next_element['calojets_px'][..., tf.newaxis],
                                       next_element['calojets_py'][..., tf.newaxis],
                                       next_element['calojets_pz'][..., tf.newaxis],
                                       next_element['calojets_eta'][..., tf.newaxis],
                                       next_element['calojets_phi'][..., tf.newaxis],
                                       next_element['calojets_energy'][..., tf.newaxis],
                                       next_element['calojets_pt'][..., tf.newaxis],), axis=-1)


            target = tf.concat((next_element['pfjets_energy'][..., tf.newaxis],
                                next_element['pfjets_px'][..., tf.newaxis],
                                next_element['pfjets_py'][..., tf.newaxis],
                                next_element['pfjets_pz'][..., tf.newaxis]), axis=-1)
            target = (target - mean_vector)/scaling_vector_t

            target = target[:, 0][..., tf.newaxis]

            num_vertices = tf.cast(tf.math.count_nonzero(next_element['tracks_pt'], axis=1), tf.float32)

            # target = next_element['pfjets_energy'][..., tf.newaxis]
            #
            # target = (target - 59.384735)/210.64401


            with tf.GradientTape() as tape:
                v = the_model(tracks_in, calo_in, num_vertices)


                # if iteration%10==0:
                    # print(np.array(v))
                    # print(np.array(target_energy))

                # print(float(v[0,0]), float(target[0,0]))
                # print(float(v[1,0]), float(target[1,0]))
                # print(float(v[2,0]), float(target[2,0]))
                print(target[0].numpy().tolist(), v[0].numpy().tolist())
                print(target[1].numpy().tolist(), v[1].numpy().tolist())
                print(target[2].numpy().tolist(), v[2].numpy().tolist())


                loss_value = tf.reduce_mean(tf.reduce_sum(tf.square(v-target), axis=-1), axis=0)
            grads = tape.gradient(loss_value, the_model.trainable_variables)

            optimizer.apply_gradients(zip(grads,the_model.trainable_variables))
            loss_value = float(loss_value)
            print(epoch, iteration, loss_value)
            tf.summary.scalar("loss", loss_value, step=iteration)
            writer.flush()
            iteration += 1


        the_model.save_weights('checkpoints/land_cruiser')
