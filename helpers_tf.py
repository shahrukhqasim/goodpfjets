import tensorflow as tf




def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def extract_fn(data_record):
  features = {
    'pfjets_px': tf.io.FixedLenFeature([], tf.string),
    'pfjets_py': tf.io.FixedLenFeature([], tf.string),
    'pfjets_pz': tf.io.FixedLenFeature([], tf.string),
    'pfjets_eta': tf.io.FixedLenFeature([], tf.string),
    'pfjets_phi': tf.io.FixedLenFeature([], tf.string),
    'pfjets_energy': tf.io.FixedLenFeature([], tf.string),
    'pfjets_pt': tf.io.FixedLenFeature([], tf.string),

    'calojets_px': tf.io.FixedLenFeature([], tf.string),
    'calojets_py': tf.io.FixedLenFeature([], tf.string),
    'calojets_pz': tf.io.FixedLenFeature([], tf.string),
    'calojets_eta': tf.io.FixedLenFeature([], tf.string),
    'calojets_phi': tf.io.FixedLenFeature([], tf.string),
    'calojets_energy': tf.io.FixedLenFeature([], tf.string),
    'calojets_pt': tf.io.FixedLenFeature([], tf.string),

    'tracks_px': tf.io.FixedLenFeature([], tf.string),
    'tracks_py': tf.io.FixedLenFeature([], tf.string),
    'tracks_pz': tf.io.FixedLenFeature([], tf.string),
    'tracks_eta': tf.io.FixedLenFeature([], tf.string),
    'tracks_phi': tf.io.FixedLenFeature([], tf.string),
    'tracks_pt': tf.io.FixedLenFeature([], tf.string),

    'ebhit_energy': tf.io.FixedLenFeature([], tf.string),
    'ebhit_eta': tf.io.FixedLenFeature([], tf.string),
    'ebhit_phi': tf.io.FixedLenFeature([], tf.string),

    'hcalhit_energy': tf.io.FixedLenFeature([], tf.string),
    'hcalhit_eta': tf.io.FixedLenFeature([], tf.string),
    'hcalhit_phi': tf.io.FixedLenFeature([], tf.string),
  }
  sample = tf.io.parse_single_example(data_record, features)
  # sample =

  sample_out = dict()
  for key, value in sample.items():
    sample_out[key] = tf.io.parse_tensor(value, out_type=tf.float32)

  return sample_out

