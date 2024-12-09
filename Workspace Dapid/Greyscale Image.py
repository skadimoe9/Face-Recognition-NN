import os
import cv2


# List of names
names = [
    "Azmira","David", "Dimas", "Fadhli", "Fadlin", "Hafidz", "Haidar", "Keiko",
    "Khansa", "Mikhael", "Raesa", "Hanna", "Puti", "Satwika", "Toni"
]

# Loop through each name and process the images
for name in names:
    input_folder = f'C:\\Users\\david\\OneDrive\\Dokumen\\1. Folder Pribadi\\Kuliah\\Semester 5\\PENGMOD\\Pengmod\\Face-Recognition-NN\\Dataset\\Foto Terpisah\\{name}'
    output_folder = f'C:\\Users\\david\\OneDrive\\Dokumen\\1. Folder Pribadi\\Kuliah\\Semester 5\\PENGMOD\\Pengmod\\Face-Recognition-NN\\Dataset\\Grey_Photo\\{name}'

    # Pastikan output_folder ada
    os.makedirs(output_folder, exist_ok=True)

    # Menelusuri semua folder dan subfolder
    for root, dirs, files in os.walk(input_folder):
        file_number = 1  # Initialize a counter for the files
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)  # Menggunakan img_path untuk membaca gambar
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    relative_path = os.path.relpath(root, input_folder)
                    output_subfolder = os.path.join(output_folder, relative_path)
                    os.makedirs(output_subfolder, exist_ok=True)

                    # Rename output file as "name_number.jpg"
                    output_file_name = f"{name}_{file_number}.jpg"
                    output_file_path = os.path.join(output_subfolder, output_file_name)  # Renaming the output file
                    cv2.imwrite(output_file_path, gray)
                    print(f"{file} berhasil dikonversi dan disimpan di {output_file_path}")

                    file_number += 1  # Increment file number for the next file
                except Exception as e:
                    print(f"{file} tidak dapat dikonversi. Error : {e}")
