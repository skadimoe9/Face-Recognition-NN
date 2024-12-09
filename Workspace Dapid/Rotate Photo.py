import os
import cv2

# List of names
names = [
    "Azmira","David", "Dimas", "Fadhli", "Fadlin", "Hafidz", "Haidar", "Keiko",
    "Khansa", "Mikhael", "Raesa", "Hanna", "Puti", "Satwika", "Toni"
]

# Fungsi untuk melakukan rotasi pada gambar
def rotate_image(image, angle):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("Angle must be 90, 180, or 270.")

# Loop through each name and process the images
for name in names:
    input_folder = f'C:\\Users\\david\\OneDrive\\Dokumen\\1. Folder Pribadi\\Kuliah\\Semester 5\\PENGMOD\\Pengmod\\Face-Recognition-NN\\Dataset\\Grey_Photo\\{name}'
    
    # Output folders for rotated images
    output_folders = {
        90: f'C:\\Users\\david\\OneDrive\\Dokumen\\1. Folder Pribadi\\Kuliah\\Semester 5\\PENGMOD\\Pengmod\\Face-Recognition-NN\\Dataset\\Rotated_Photo\\{name}_90',
        180: f'C:\\Users\\david\\OneDrive\\Dokumen\\1. Folder Pribadi\\Kuliah\\Semester 5\\PENGMOD\\Pengmod\\Face-Recognition-NN\\Dataset\\Rotated_Photo\\{name}_180',
        270: f'C:\\Users\\david\\OneDrive\\Dokumen\\1. Folder Pribadi\\Kuliah\\Semester 5\\PENGMOD\\Pengmod\\Face-Recognition-NN\\Dataset\\Rotated_Photo\\{name}_270'
    }

    # Pastikan output folder ada untuk setiap rotasi
    for angle in output_folders.values():
        os.makedirs(angle, exist_ok=True)

    # Menelusuri semua folder dan subfolder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)  # Menggunakan img_path untuk membaca gambar

                    # Rotasi untuk 90, 180, dan 270 derajat
                    for angle in [90, 180, 270]:
                        rotated_img = rotate_image(img, angle)

                        # Menyimpan gambar yang sudah dirotasi ke folder yang sesuai
                        relative_path = os.path.relpath(root, input_folder)
                        output_subfolder = os.path.join(output_folders[angle], relative_path)
                        os.makedirs(output_subfolder, exist_ok=True)

                        # Menentukan path output untuk gambar yang sudah dirotasi
                        output_file_path = os.path.join(output_subfolder, f"{file.split('.')[0]}_rotated_{angle}.{file.split('.')[1]}")
                        cv2.imwrite(output_file_path, rotated_img)
                        print(f"{file} berhasil dirotasi {angle} derajat dan disimpan di {output_file_path}")

                except Exception as e:
                    print(f"{file} tidak dapat diproses. Error: {e}")
