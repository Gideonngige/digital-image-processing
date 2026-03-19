# import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# Load a differenet build-in test image
original = data.coins()
print('Image shape:', original.shape)

# function to add salt and pepper
def add_salt_pepper(image, noise_level=0.05):
    """Add salt-and-pepper noise to a greyscale image."""
    noisy = image.copy()
    total_pixels = image.size
    n_corrupt = int(total_pixels * noise_level)

    # Add SALT
    salt_coords = [np.random.randint(0, i, n_corrupt // 2) for i in image.shape]

    noisy[salt_coords[0], salt_coords[1]] = 255
    # Add PEPPER
    pepper_coords = [np.random.randint(0, i, n_corrupt // 2) for i in image.shape]
    noisy[pepper_coords[0], pepper_coords[1]] = 0
    return noisy

# Apply the function: 5% of the pixels will be corrupted
noisy = add_salt_pepper(original, noise_level=0.05)
print('Noise added. Corrupted pixels:', int(original.size * 0.05))

# Apply median filter with 3x3 kernel
denoised_median = cv2.medianBlur(noisy, ksize=3)

denoised_gaussian = cv2.GaussianBlur(noisy,(3,3),sigmaX=1)

print('Median filter applied successfully.')

# Display and save the results
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('Noise Removal: Median vs Gaussian Filter', fontsize=14, fontweight='bold')

axes[0].imshow(original, cmap='gray')
axes[0].set_title('1. Original Image', fontsize=11)
axes[0].axis('off')

axes[1].imshow(noisy, cmap='gray')
axes[1].set_title('2. Noisy Image (5% salt & pepper)', fontsize=11)
axes[1].axis('off')

axes[2].imshow(denoised_median, cmap='gray')
axes[2].set_title('3. Median Filter (3x3)', fontsize=11)
axes[2].axis('off')

axes[3].imshow(denoised_gaussian, cmap='gray')
axes[3].set_title('4. Gaussian Blur (3x3)', fontsize=11)
axes[3].axis('off')

plt.tight_layout()
plt.savefig('exercise2_output.png', dpi=150, bbox_inches='tight')
plt.show()

print('Figure saved as exercise2_output.png')
