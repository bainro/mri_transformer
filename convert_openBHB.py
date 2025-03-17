# @TODO extract labels!

import os
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from cupyx.scipy.ndimage import zoom 
from estimator import FeatureExtractor
from problem import get_train_data, get_test_data, DatasetHelper

os.environ["VBM_MASK"] = "./cat12vbm_space-MNI152_desc-gm_TPM.nii.gz"
os.environ["QUASIRAW_MASK"] = "./quasiraw_space-MNI152_desc-brain_T1w.nii.gz"

# Create output directory
directory = "subjects"
os.makedirs(directory, exist_ok=True)
directory = "subjects/train"
os.makedirs(directory, exist_ok=True)
directory = "subjects/test"
os.makedirs(directory, exist_ok=True)

for k, data_loader in enumerate([get_train_data, get_test_data]):
    data = DatasetHelper(data_loader=data_loader)
    X, y = data.get_data()

    directory = "subjects/train"
    if k == 1: # overwrite
        directory = "subjects/test"        

    # Define batch size
    BATCH_SIZE = 100  # Adjust this based on available memory
    num_samples = X.shape[0]

    # Target resolution
    x_t, y_t, z_t = 128, 128, 128

    resized_T1 = None
    # Process data in batches
    for start in range(0, num_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_samples)
        print(f"Processing subjects {start} to {end}...")

        # Load and extract a batch
        X_batch = FeatureExtractor(dtype="quasiraw").transform(X[start:end])

        # Get the real resolution of the batch
        x_r, y_r, z_r = X_batch.shape[-3:]
        scales = (1, x_t / x_r, y_t / y_r, z_t / z_r)

        # Resize and save each subject
        for i in range(X_batch.shape[0]):
            subject_idx = start + i
            resized_T1 = zoom(cp.array(X_batch[i]), scales, order=1)
            npy_name = os.path.join(directory, f'{subject_idx:04d}_T1.npy')
            np.save(npy_name, cp.asnumpy(resized_T1))

    # Plot one sample for verification
    resized_T1 = np.expand_dims(cp.asnumpy(resized_T1), axis=0)
    data.plot_data(resized_T1, sample_id=0, channel_id=0)
    plt.show()
