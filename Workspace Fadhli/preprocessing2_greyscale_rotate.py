import os
from PIL import Image

# Set the input and output directory paths
input_directory = "./Dataset/Foto Terpisah"
output_directory = "./Dataset/Foto Greyscale"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created output directory: {output_directory}")
else:
    print(f"Output directory already exists: {output_directory}")

# List all folders in the input directory
folders = [f for f in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, f))]
print(f"Found folders: {folders}")

# Process each folder and photo
for folder in folders:
    folder_path = os.path.join(input_directory, folder)
    output_folder_path = os.path.join(output_directory, folder)
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f"Created output folder: {output_folder_path}")
    
    photo_counter = 1
    for photo in os.listdir(folder_path):
        if photo.endswith(('.png', '.jpg', '.jpeg')):
            photo_path = os.path.join(folder_path, photo)
            image = Image.open(photo_path)
            
            # Convert to greyscale
            image = image.convert("L")
            
            # Rotate the image and save with the specified naming convention
            for angle in [0, 90, 180, 270]:
                rotated_image = image.rotate(angle)
                output_photo_path = os.path.join(output_folder_path, f"{folder}_{photo_counter}.jpg")
                rotated_image.save(output_photo_path)
                print(f"Saved rotated image: {output_photo_path}")
                photo_counter += 1

print("Processing complete.")