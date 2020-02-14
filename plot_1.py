import tensorflow as tf
import helpers_tf as htf
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


batch_size = 512

train_set = tf.data.TFRecordDataset(['train.tfrecords'], "GZIP")
train_set = train_set.map(htf.extract_fn)



vv = 0

show_no =1
for next_element in train_set:

    if vv < show_no:
        vv+= 1
        continue


    tracks_px = next_element['tracks_px'].numpy()
    tracks_py = next_element['tracks_py'].numpy()
    tracks_pz = next_element['tracks_pz'].numpy()

    calojet = np.array([next_element['calojets_px'].numpy(), next_element['calojets_py'].numpy(), next_element['calojets_pz'].numpy()])
    calojet = calojet / np.sqrt(np.sum(calojet**2))

    pfjet = np.array([next_element['pfjets_px'].numpy(), next_element['pfjets_py'].numpy(), next_element['pfjets_pz'].numpy()])
    pfjet= pfjet / np.sqrt(np.sum(pfjet**2))

    widths = np.concatenate([tracks_px[..., np.newaxis], tracks_py[..., np.newaxis], tracks_pz[..., np.newaxis]])
    widths = np.sum(widths ** 2, axis=-1)
    # widths = np.amin(2, np.amax(0.1, 2 * (widths - np.mean(widths)) / np.var(widths)))
    widths = np.minimum(2, np.maximum(0.1, (widths - np.mean(widths)) / np.var(widths) + 1))


    ax = Axes3D(plt.figure())
    for i in range(len(tracks_px)):
        PX = tracks_px[i]
        PY = tracks_py[i]
        PZ = tracks_pz[i]

        norm = np.sqrt(PX**2 + PY**2 + PZ**2)

        PX = PX / norm
        PY = PY / norm
        PZ = PZ / norm

        if tracks_px[i] == 0 and tracks_py[i] == 0 and tracks_pz[i] == 0:
            continue
        #
        # if tracks_px[i]**2 + tracks_py[i]**2 + tracks_pz[i]**2 > 10:
        #     continue

        print("Plotting", tracks_px[i], tracks_py[i], tracks_pz[i])
        ax.plot([0, PX], [0, PY], zs=[0, PZ], color='red', linewidth=widths[i])

    ax.plot([0, calojet[0]], [0, calojet[1]], zs=[0, calojet[2]], color='blue', linewidth=3)
    ax.plot([0, pfjet[0]], [0, pfjet[1]], zs=[0, pfjet[2]], color='green', linewidth=3)

    plt.show()
    input()


