import os
from PIL import Image, ExifTags

def exif_transpose(image):
    """
    Apply the EXIF orientation to the image.
    """
    try:
        exif = image._getexif()
        if exif is not None:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(exif.items())
            orientation = exif.get(orientation, None)

            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # Cases: image don't have getexif
        pass
    return image

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
            
            # Apply EXIF orientation
            image = exif_transpose(image)
            
            # Convert to greyscale
            image = image.convert("L")
            
            # Save the greyscale image with the specified naming convention
            output_photo_path = os.path.join(output_folder_path, f"{folder}_{photo_counter}.jpg")
            image.save(output_photo_path)
            print(f"Saved image: {output_photo_path}")
            photo_counter += 1

print("Processing complete.")