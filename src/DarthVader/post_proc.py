import os
from PIL import Image
import numpy as np

def crop_black_region(image):
    # Convert image to numpy array
    image_array = np.array(image)
    
    # Find all rows and columns where the pixels are not black
    non_black_pixels = np.where(image_array[:, :, :3] != 0)
    
    # Calculate bounding box
    if len(non_black_pixels[0]) > 0:  # Check if there's non-black content
        top, left = np.min(non_black_pixels[0]), np.min(non_black_pixels[1])
        bottom, right = np.max(non_black_pixels[0]), np.max(non_black_pixels[1])
        
        # Crop the image
        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image
    else:
        return image  # Return original image if fully black

def process_folder(folder_path):
    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(subdir, file)
                
                # Load image
                with Image.open(file_path) as img:
                    # Crop black regions
                    cropped_img = crop_black_region(img)
                    
                    # Save the cropped image, overwriting the original
                    cropped_img.save(file_path)

# Specify the main folder containing subfolders I1, I2, etc.


