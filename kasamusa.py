import numpy as np
import matplotlib.pyplot as plt

# Generate two 64x64 Gaussian white noise images
noise1 = np.random.normal(loc=0, scale=1, size=(64, 64))
noise2 = np.random.normal(loc=0, scale=1, size=(64, 64))

# Display the noise images
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(noise1, cmap='hot')
axes[0].set_title("Gaussian White Noise 1")
axes[0].axis("off")

axes[1].imshow(noise2, cmap='hot')
axes[1].set_title("Gaussian White Noise 2")
axes[1].axis("off")

plt.show()