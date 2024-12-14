import os
from PIL import Image

# Set the input and output directory paths
input_directory = "./Dataset/Foto Greyscale Rotate"
output_directories = {
    "30x30": "./Dataset/Foto_Resize_Rotate_30x30",
    "50x50": "./Dataset/Foto_Resize_Rotate_50x50",
    "70x70": "./Dataset/Foto_Resize_Rotate_70x70"
}

# Create the output directories if they don't exist
for size, output_directory in output_directories.items():
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
    
    # Sort the list of photos to ensure consistent order
    photos = sorted([p for p in os.listdir(folder_path) if p.endswith(('.png', '.jpg', '.jpeg'))])
    
    for photo in photos:
        photo_path = os.path.join(folder_path, photo)
        image = Image.open(photo_path)
        
        # Resize the image and save in the respective output folders
        for size, output_directory in output_directories.items():
            width, height = map(int, size.split('x'))
            resized_image = image.resize((width, height), Image.Resampling.LANCZOS)
            
            output_folder_path = os.path.join(output_directory, folder)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
                print(f"Created output folder: {output_folder_path}")
            
            # Extract the original photo number from the filename
            photo_number = photo.split('_')[-1].split('.')[0]
            output_photo_path = os.path.join(output_folder_path, f"{folder}_{size}_{photo_number}.jpg")
            resized_image.save(output_photo_path)
            print(f"Saved resized image: {output_photo_path}")

print("Processing complete.")