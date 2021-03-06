import h5py
from JetDataset import JetDataset
from visualizations import visualize
import numpy as np
import tensorflow as tf
import helpers_tf as htf
import matplotlib.pyplot as plt

index = 3


# filename = '/eos/home-s/sqasim/Datasets/PF_alpha/step3_750_800hlt_patatrack.h5'
# filename = '/eos/home-s/sqasim/Datasets/PF_alpha/step3_800_850hlt_patatrack.h5'
# filename = '/eos/home-s/sqasim/Datasets/PF_alpha/step3_900_950hlt_patatrack.h5'


filename = '/afs/cern.ch/work/s/sqasim/temp/step3_750_800hlt_patatrack.h5'


data = JetDataset(filename)


def serialize_example(pfjets_px, pfjets_py, pfjets_pz, pfjets_eta, pfjets_phi, pfjets_energy, pfjets_pt, \
    calojets_px, calojets_py, calojets_pz, calojets_eta, calojets_phi, calojets_energy, calojets_pt,  \
    tracks_px, tracks_py, tracks_pz, tracks_eta, tracks_phi, tracks_pt, ebhit_energy, ebhit_eta, ebhit_phi, hcalhit_energy, hcalhit_eta, hcalhit_phi):

    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
      'pfjets_px': htf._bytes_feature(tf.io.serialize_tensor(pfjets_px)), # 800
      'pfjets_py': htf._bytes_feature(tf.io.serialize_tensor(pfjets_py)),
      'pfjets_pz': htf._bytes_feature(tf.io.serialize_tensor(pfjets_pz)),
      'pfjets_eta':htf. _bytes_feature(tf.io.serialize_tensor(pfjets_eta)),
      'pfjets_phi':htf. _bytes_feature(tf.io.serialize_tensor(pfjets_phi)),
      'pfjets_energy':htf. _bytes_feature(tf.io.serialize_tensor(pfjets_energy)),
      'pfjets_pt':htf. _bytes_feature(tf.io.serialize_tensor(pfjets_pt)),

      'calojets_px':htf. _bytes_feature(tf.io.serialize_tensor(calojets_px)),
      'calojets_py':htf. _bytes_feature(tf.io.serialize_tensor(calojets_py)),
      'calojets_pz':htf. _bytes_feature(tf.io.serialize_tensor(calojets_pz)),
      'calojets_eta':htf. _bytes_feature(tf.io.serialize_tensor(calojets_eta)),
      'calojets_phi':htf. _bytes_feature(tf.io.serialize_tensor(calojets_phi)),
      'calojets_energy':htf. _bytes_feature(tf.io.serialize_tensor(calojets_energy)),
      'calojets_pt':htf. _bytes_feature(tf.io.serialize_tensor(calojets_pt)),

      'tracks_px':htf. _bytes_feature(tf.io.serialize_tensor(tracks_px)),
      'tracks_py':htf. _bytes_feature(tf.io.serialize_tensor(tracks_py)),
      'tracks_pz':htf. _bytes_feature(tf.io.serialize_tensor(tracks_pz)),
      'tracks_eta':htf. _bytes_feature(tf.io.serialize_tensor(tracks_eta)),
      'tracks_phi':htf. _bytes_feature(tf.io.serialize_tensor(tracks_phi)),
      'tracks_pt':htf. _bytes_feature(tf.io.serialize_tensor(tracks_pt)),
      'ebhit_energy':htf. _bytes_feature(tf.io.serialize_tensor(ebhit_energy)),
      'ebhit_eta':htf. _bytes_feature(tf.io.serialize_tensor(ebhit_eta)),
      'ebhit_phi':htf. _bytes_feature(tf.io.serialize_tensor(ebhit_phi)),
      'hcalhit_energy':htf. _bytes_feature(tf.io.serialize_tensor(hcalhit_energy)),
      'hcalhit_eta':htf. _bytes_feature(tf.io.serialize_tensor(hcalhit_eta)),
      'hcalhit_phi':htf. _bytes_feature(tf.io.serialize_tensor(hcalhit_phi)),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_to_tfrecords(initial, final, filename):
    count = 0
    count2 = 0

    pixel_tracks_count = []
    ecal_hits_count = []
    hcal_hits_count = []


    options = tf.io.TFRecordOptions("GZIP")
    with tf.io.TFRecordWriter(filename, options=options) as writer:
        for idx in range(initial, final):
            print("I", idx)
            event = data.getitemex(idx)
            num_jets = np.count_nonzero(event[5])

            # print("N", num_jets)

            # count2 += num_jets
            # count2 += num_jets
            for idx2 in range(num_jets):
                sum1 = np.sum(event[-7][idx2])
                sum2 = np.sum(event[-3][idx2])
                sum3 = np.sum(event[-6][idx2])

                # pixel_tracks_count += [np.count_nonzero(event[-7][idx2])]
                # ecal_hits_count += [np.count_nonzero(event[-6][idx2])]
                # hcal_hits_count += [np.count_nonzero(event[-3][idx2])]




                if sum1!=0 or sum2!=0 or sum3!=0:
                    print(np.count_nonzero(event[-7][idx2]), np.count_nonzero(event[-6][idx2]), np.count_nonzero(event[-3][idx2]))
                    # continue
                    print(sum1, sum2, sum3)

                    count2 += 1

                    pixel_tracks_count += [np.count_nonzero(event[-7][idx2])]
                    ecal_hits_count += [np.count_nonzero(event[-6][idx2])]
                    hcal_hits_count += [np.count_nonzero(event[-3][idx2])]

                    continue
                    print(float(tf.reduce_sum(event[-3][idx2])),
                          float(tf.reduce_sum(event[-6][idx2])),
                          float(tf.reduce_sum(event[-7][idx2])),
                          float(tf.reduce_sum(event[3][idx2])),
                          float(tf.reduce_sum(event[11][idx2])),
                          float(tf.reduce_sum(event[12][idx2])))
                    example = serialize_example(*(x[idx2] for x in event))
                    count+=1
                    writer.write(example)

    print("Count is ", count2)

    pixel_tracks_count = np.array(pixel_tracks_count)
    ecal_hits_count = np.array(ecal_hits_count)
    hcal_hits_count = np.array(hcal_hits_count)

    print("Pixel tracks",np.histogram(pixel_tracks_count, bins=[0, 1,10,30,100,1000]))
    print("ECal hits",np.histogram(ecal_hits_count, bins=[0, 1,10,30,100,1000]))
    print("HCal hits",np.histogram(hcal_hits_count, bins=[0, 1,10,30,100,1000]))

    print("Pixel tracks",np.histogram(pixel_tracks_count[pixel_tracks_count>0], bins=[1,2,3,4,5,6,7,8,9,10,13,16,20,25,50,101]))
    print("ECal hits",np.histogram(ecal_hits_count[ecal_hits_count>0], bins=[1,2,3,4,5,6,7,8,9,10,13,16,20,25,50,1000]))
    print("HCal hits",np.histogram(hcal_hits_count[hcal_hits_count>0]))

    np.savetxt('dump.txt',ecal_hits_count)



    print(len(pixel_tracks_count[pixel_tracks_count>0]))
    print(len(ecal_hits_count[ecal_hits_count>0]))
    print(len(hcal_hits_count[hcal_hits_count>0]))

    0/0

    plt.hist(pixel_tracks_count, range=None, bins=[0, 1,10,30,100,1000], histtype='bar', color='green')
    plt.hist(ecal_hits_count, range=None, bins=[0, 1,10,30,100,1000], histtype='bar', color='red')
    plt.hist(hcal_hits_count, range=None, bins=[0, 1,10,30,100,1000], histtype='bar', color='blue')

    plt.legend(['Pixel tracks', 'ECal hits', 'HCal hits'])
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title('Distribution of data')
    # plt.xticks([0,1,10,30,100])
    plt.savefig('blabla.pdf')

    0/0

    print("Written", count, "records to",filename)



# write_to_tfrecords(0, 2000, '/eos/home-s/sqasim/Datasets/PF_alpha/tf/train.tfrecords')
# write_to_tfrecords(2000, 2250, '/eos/home-s/sqasim/Datasets/PF_alpha/tf/test.tfrecords')
# write_to_tfrecords(2250, 2500, '/eos/home-s/sqasim/Datasets/PF_alpha/tf/val.tfrecords')


# write_to_tfrecords(0, 2000, '/eos/home-s/sqasim/Datasets/PF_alpha/tf/train.tfrecords')
# write_to_tfrecords(2000, 2250, '/eos/home-s/sqasim/Datasets/PF_alpha/tf/test.tfrecords')
write_to_tfrecords(0, 2500, '/eos/home-s/sqasim/Datasets/PF_alpha/tf/val3.tfrecords')

