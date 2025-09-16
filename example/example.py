import cv2
import torch
import matplotlib.pyplot as plt

import jbu_cuda


radius = 2 # Filter radius
sigma_s = 0.5 # Spatial sigma
p_s = 8 # Downsampling factor


# Load the original image
original_image = cv2.imread('example_aerial.png')

# converting BGR to RGB
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Downsample the image
downsampled_image = cv2.resize(original_image, (original_image.shape[1] // p_s, original_image.shape[0] // p_s), interpolation=cv2.INTER_LANCZOS4)

guidance = torch.tensor(cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)).unsqueeze(0).unsqueeze(0) / 255.0
sigma_r = guidance.std(dim=(2,3))

result:torch.Tensor = jbu_cuda.upsample(guidance.float().to('cuda') , torch.tensor(downsampled_image).permute(2,0,1).unsqueeze(0).to('cuda') / 255.0, radius, sigma_s, sigma_r)
torch.cuda.empty_cache()

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

# Display each image in a subplot
axs[0].imshow(guidance[0, 0])
axs[0].set_title('Guidance Image')
axs[0].axis('off')  # Hide axes

axs[1].imshow(downsampled_image)
axs[1].set_title('Downsampled Image')
axs[1].axis('off')  # Hide axes

axs[2].imshow(result.cpu()[0].permute(1,2,0))
axs[2].set_title('Result Image (JBU)')
axs[2].axis('off')  # Hide axes


# Show the plot
plt.tight_layout()
plt.show()
