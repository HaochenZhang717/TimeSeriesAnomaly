from generation_models import FM_TS
from Trainers import FlowTSPretrain
from dataset_utils import ECGDataset

import argparse
import torch


import matplotlib.pyplot as plt


all_data = torch.load("./samples_path/flow/mitdb106v-2025-11-30-03:01:14/all_data.pt", map_location="cpu")



# plt.plot(all_data["orig_normal_train_signal"][0,:,0])
# plt.plot(all_data["orig_normal_train_signal"][1000,:,0])
# plt.plot(all_data["orig_normal_train_signal"][2000,:,0])
# plt.plot(all_data["orig_normal_train_signal"][3000,:,0])
# plt.plot(all_data["orig_normal_train_signal"][4000,:,0])
# plt.title("Original Normal Signal Examples")
# plt.show()
#
# plt.plot(all_data["orig_anomaly_train_signal"][0,:,0], label="signal")
# plt.plot(all_data["orig_anomaly_train_label"][0,:], label="label")
# plt.title("original anomaly train signal")
# plt.show()
#
#
# plt.plot(all_data["orig_anomaly_train_signal"][100,:,0], label="signal")
# plt.plot(all_data["orig_anomaly_train_label"][100,:], label="label")
# plt.title("original anomaly train signal")
# plt.show()
#
#
#
# plt.plot(all_data["orig_anomaly_train_signal"][200,:,0], label="signal")
# plt.plot(all_data["orig_anomaly_train_label"][200,:], label="label")
# plt.title("original anomaly train signal")
# plt.show()

plt.plot(all_data["all_samples"][200,:,0], label="signal")
plt.plot(all_data["all_anomaly_labels"][200,:], label="label")
plt.title("generated anomaly")
plt.show()


plt.plot(all_data["all_samples"][0,:,0], label="signal")
plt.plot(all_data["all_anomaly_labels"][0,:], label="label")
plt.title("generated anomaly")
plt.show()


plt.plot(all_data["all_samples"][10,:,0], label="signal")
plt.plot(all_data["all_anomaly_labels"][10,:], label="label")
plt.title("generated anomaly")
plt.show()