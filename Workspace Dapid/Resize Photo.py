import os
import cv2

# List of names
names = [
    "David", "Dimas", "Fadhli", "Fadlin", "Hafidz", "Haidar", "Keiko",
    "Khansa", "Mikhael", "Raesa", "Toni", "Hanna", "Puti"
]

# Fungsi untuk resize gambar ke ukuran 70x70
def resize_image(image, size=(70, 70)):
    return cv2.resize(image, size)

# Loop through each name and process the images
for name in names:
    input_folder = f'C:\\Users\\david\\OneDrive\\Dokumen\\1. Folder Pribadi\\Kuliah\\Semester 5\\PENGMOD\\Pengmod\\Face-Recognition-NN\\Dataset\\Grey_Photo\\{name}'
    output_folder = f'C:\\Users\\david\\OneDrive\\Dokumen\\1. Folder Pribadi\\Kuliah\\Semester 5\\PENGMOD\\Pengmod\\Face-Recognition-NN\\Dataset\\Resized_Photo\\{name}_70x70'

    # Pastikan output folder ada
    os.makedirs(output_folder, exist_ok=True)

    # Menelusuri semua folder dan subfolder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)  # Menggunakan img_path untuk membaca gambar
                    resized_img = resize_image(img, size=(70, 70))  # Resize gambar ke 70x70

                    # Menyimpan gambar yang sudah diresize
                    relative_path = os.path.relpath(root, input_folder)
                    output_subfolder = os.path.join(output_folder, relative_path)
                    os.makedirs(output_subfolder, exist_ok=True)

                    # Menentukan path output untuk gambar yang sudah diresize
                    output_file_path = os.path.join(output_subfolder, f"{file.split('.')[0]}_resized_70x70.{file.split('.')[1]}")
                    cv2.imwrite(output_file_path, resized_img)
                    print(f"{file} berhasil diresize dan disimpan di {output_file_path}")

                except Exception as e:
                    print(f"{file} tidak dapat diproses. Error: {e}")
