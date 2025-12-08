import torch
import matplotlib.pyplot as plt



results = torch.load("generated_anomaly.pt", map_location='cpu')


for i in range(768):
    plt.plot(results['all_samples'][i, :, 0], label="generated")
    plt.plot(results['all_real'][i, :, 0], label="real")
    plt.plot(results['all_anomaly_labels'][i], label="anomaly labels")
    plt.legend()
    plt.show()
