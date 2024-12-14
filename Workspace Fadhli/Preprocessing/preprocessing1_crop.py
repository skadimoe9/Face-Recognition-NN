import os
from PIL import Image

# Set the input and output directory paths
input_directory = "./Dataset/Foto Terpisah/Satwika"
output_directory = "./Dataset/Foto_Sebanding/Satwika"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created output directory: {output_directory}")
else:
    print(f"Output directory already exists: {output_directory}")

# Function to resize and crop the image to 1:1 ratio
def resize_and_crop(image, size):
    img_ratio = image.width / float(image.height)
    target_ratio = size[0] / float(size[1])
    
    if target_ratio > img_ratio:
        image = image.resize((size[0], int(round(size[0] * image.height / image.width))), Image.Resampling.LANCZOS)
    elif target_ratio < img_ratio:
        image = image.resize((int(round(size[1] * image.width / image.height)), size[1]), Image.Resampling.LANCZOS)
    else:
        image = image.resize((size[0], size[1]), Image.Resampling.LANCZOS)
    
    width, height = image.size
    left = (width - size[0]) / 2
    top = (height - size[1]) / 2 - (0.2 * height)  # Move the crop area slightly higher
    right = (width + size[0]) / 2
    bottom = (height + size[1]) / 2 - (0.2 * height)

    return image.crop((left, top, right, bottom))

# Process each photo in the Satwika folder
for photo in os.listdir(input_directory):
    if photo.endswith(('.png', '.jpg', '.jpeg')):
        photo_path = os.path.join(input_directory, photo)
        image = Image.open(photo_path)
        
        # Resize and crop the image to 1:1 ratio focusing on the center
        resized_image = resize_and_crop(image, (image.width, image.width))
        output_photo_path = os.path.join(output_directory, f"{os.path.splitext(photo)[0]}_center.jpg")
        resized_image.save(output_photo_path)
        print(f"Saved resized image: {output_photo_path}")

print("Processing complete.")