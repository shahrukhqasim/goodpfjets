import tensorflow as tf
import helpers_tf as htf
import numpy as np
from model_3 import SimplerGraphModel
import matplotlib.pyplot as plt

# batch_size = 16
#
# train_set = tf.data.TFRecordDataset(['train.tfrecords'], "GZIP")
# train_set = train_set.map(htf.extract_fn)
# train_set = train_set.shuffle(buffer_size=5000)
# train_set = train_set.batch(batch_size)
#
#
# mean_vector = np.array([59.384735, 0.0053087026, -0.0077821156, -0.5366358])[np.newaxis, ...]
# mean_vector_t = tf.cast(tf.constant(mean_vector), tf.float32)
#
# scaling_vector = np.array([210.64401, 10.853447, 11.070456, 218.28716])[np.newaxis, ...]
# scaling_vector_t = tf.cast(tf.constant(scaling_vector), tf.float32)
#
#
#
# # next_element = iterator.get_next()
# val_set = tf.data.TFRecordDataset(['val.tfrecords'], "GZIP")
# val_set = val_set.map(htf.extract_fn)
# val_set = val_set.batch(batch_size)
#
# test_set = tf.data.TFRecordDataset(['test.tfrecords'], "GZIP")
#
#
#
# the_model = SimplerGraphModel(num_outputs=1)
# the_model.call(np.zeros((batch_size, 100, 6)),np.zeros((batch_size, 1000, 3)),np.zeros((batch_size, 2000, 3)), np.zeros((batch_size, 7)), np.ones((batch_size,)), np.ones((batch_size,)), np.ones((batch_size,)))
#
#
#
# if True:
#     the_model.load_weights('checkpoints/land_cruiser')
#
#
# pf_energies = []
# calo_energies = []
# predicted_energies = []
#
# nonzeros=[]
#
# num = 0
#
#
# for next_element in val_set:
#     #
#     # if num==150:
#     #     break
#
#     if len(next_element['tracks_px']) < batch_size:
#         continue
#     tracks_in = tf.concat((next_element['tracks_px'][..., tf.newaxis],
#                            next_element['tracks_py'][..., tf.newaxis],
#                            next_element['tracks_pz'][..., tf.newaxis],
#                            next_element['tracks_eta'][..., tf.newaxis],
#                            next_element['tracks_phi'][..., tf.newaxis],
#                            next_element['tracks_pt'][..., tf.newaxis],), axis=-1)
#
#     calo_in = tf.concat((next_element['calojets_px'][..., tf.newaxis],
#                          next_element['calojets_py'][..., tf.newaxis],
#                          next_element['calojets_pz'][..., tf.newaxis],
#                          next_element['calojets_eta'][..., tf.newaxis],
#                          next_element['calojets_phi'][..., tf.newaxis],
#                          next_element['calojets_energy'][..., tf.newaxis],
#                          next_element['calojets_pt'][..., tf.newaxis],), axis=-1)
#
#     target = tf.concat((next_element['pfjets_energy'][..., tf.newaxis],
#                         next_element['pfjets_px'][..., tf.newaxis],
#                         next_element['pfjets_py'][..., tf.newaxis],
#                         next_element['pfjets_pz'][..., tf.newaxis]), axis=-1)
#
#     ecal_in = tf.concat((next_element['ebhit_energy'][..., tf.newaxis],
#                          next_element['ebhit_eta'][..., tf.newaxis],
#                          next_element['ebhit_phi'][..., tf.newaxis],), axis=-1)
#
#     hcal_in = tf.concat((next_element['hcalhit_energy'][..., tf.newaxis],
#                          next_element['hcalhit_eta'][..., tf.newaxis],
#                          next_element['hcalhit_phi'][..., tf.newaxis],), axis=-1)
#
#     # target = target* scaling_vector_t + mean_vector_t
#
#     # target = (target[:, 0][..., tf.newaxis] - 30) * scaling_vector_t + mean_vector
#     num_vertices_tracks = tf.cast(tf.math.count_nonzero(next_element['tracks_pt'], axis=1), tf.float32)
#     num_vertices_ecal = tf.cast(tf.math.count_nonzero(next_element['ebhit_energy'], axis=1), tf.float32)
#     num_vertices_hcal = tf.cast(tf.math.count_nonzero(next_element['hcalhit_energy'], axis=1), tf.float32)
#
#
#     # nonzeros += num_vertices.numpy().tolist()
#     v = the_model(tracks_in, hcal_in, ecal_in, calo_in, num_vertices_tracks, num_vertices_ecal, num_vertices_hcal)
#
#
#     # v = v* scaling_vector_t + mean_vector_t
#
#
#     calo_energy = calo_in[:,5].numpy()
#     pf_energy = target[:,0].numpy()
#
#     predicted_energy = v[:,0].numpy() * calo_energy
#
#     calo_energy = calo_energy.tolist()
#     pf_energy = pf_energy.tolist()
#
#     predicted_energy = predicted_energy.tolist()
#
#     print(predicted_energy)
#     print(calo_energy)
#
#
#     pf_energies += pf_energy
#     calo_energies += calo_energy
#     predicted_energies += predicted_energy
#
#     print(num)
#     num+=1
#
#
#
#
# print(pf_energies)
# print(calo_energies)
# print(predicted_energies)
#


# pf_energies = np.array(pf_energies)
# calo_energies = np.array(calo_energies)
# predicted_energies = np.array(predicted_energies)


# np.savetxt('pf.txt',pf_energies)
# np.savetxt('calo.txt',calo_energies)
# np.savetxt('pred.txt',predicted_energies)

pf_energies = np.loadtxt('pf.txt')
calo_energies = np.loadtxt('calo.txt')
predicted_energies = np.loadtxt('pred.txt')


plt.hist(pf_energies/calo_energies, range=(0.1, 3), bins=20, histtype='step', color='green')
plt.hist(pf_energies/predicted_energies, range=(0.1, 3), bins=20, histtype='step', color='red')

plt.legend(['pf/calo','pf/predicted'])
# plt.title('Calo energies histogram')
plt.show()


plt.hist((pf_energies-calo_energies)**2, range=(0, 10000), bins=[0,300, 600, 1000, 2000, 4000, 10000], histtype='step', color='green')
plt.hist((pf_energies-predicted_energies)**2, range=(0, 10000), bins=[0,300, 600, 1000, 2000, 4000, 10000], histtype='step', color='red')

plt.legend(['(pf-calo)^2','(pf-predicted)^2'])
# plt.title('Calo energies histogram')
plt.show()


plt.scatter(x=pf_energies, y=(calo_energies), color='green', s=0.7)
plt.scatter(x=pf_energies, y=(predicted_energies), color='red', s=0.7)
plt.xlabel('Pf Energy')
plt.ylabel('calo or predicted energy (z)')
plt.legend(['z = calo energy','z = predicted energy'])

plt.plot([0, 1750], [0, 1750], color='black')

plt.show()

print(pf_energies)
print(predicted_energies)

print(np.mean(nonzeros))


