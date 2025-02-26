import numpy as np
import matplotlib.pyplot as plt

def create_radial_constant_image(size, mean=0, stddev=1, contrast_factor=2):
    # Create a 2D grid of coordinates
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    r = np.round(np.sqrt(x**2 + y**2),3)
    radial_values = {}
    for i in np.unique(r):
        radial_values[i] = np.random.normal(loc=mean, scale=stddev)
    constant_radial_image = np.vectorize(lambda x: radial_values[x])(r)
    constant_radial_image = (constant_radial_image - np.min(constant_radial_image)) / (np.max(constant_radial_image) - np.min(constant_radial_image))
    constant_radial_image = np.clip((constant_radial_image - 0.5) * contrast_factor + 0.5, 0, 1)
    return constant_radial_image

# Image size
image_size = 256

# Create a radial constant image with increased contrast
radial_constant_image = create_radial_constant_image(image_size)

# Plot the image
plt.imshow(radial_constant_image, cmap='gray')
plt.title("Radial Constant Image with Increased Contrast")
plt.colorbar()
plt.show()
