import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

def add_salt_pepper(image, noise_level=0.03):
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

original = data.camera()

# Step 1: Add Salt and Pepper Noise
noisy = add_salt_pepper(original, noise_level=0.03)

# Step 2: Apply Gaussian blur on top (simulate camers shake/ defocus)
degraded = cv2.GaussianBlur(noisy,(5,5),sigmaX=1.5)

print('Degraded image ready.')

# Stage 1: Median filter to remove noise
denoised = cv2.medianBlur(degraded, ksize=3)

# Stage 2: Laplacian sharpening on the clean denoised image
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
laplacian = cv2.filter2D(denoised.astype(np.float32), -1, kernel)
restored = np.clip(denoised.astype(np.float32) + laplacian, 0, 255).astype(np.uint8)

print('Restoration complete.')

# Display and save the results
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Full Restoration Pipeline: Denoise → Sharpen', fontsize=14, fontweight='bold')

images = [original, degraded, denoised, restored]
titles = ['1. Original', '2. Degraded (Noisy + Blurred)', '3. Denoised (Median filter)', '4. Restored (Denoise + Sharpened)']

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title, fontsize=11)
    ax.axis('off')

plt.tight_layout()
plt.savefig('exercise3_output.png', dpi=150, bbox_inches='tight')
plt.show()
