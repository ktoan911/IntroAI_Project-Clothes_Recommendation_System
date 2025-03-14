import os
import random

# Thay đổi đường dẫn này thành thư mục chính chứa các folder con
root_folder = r"D:\Python\AI_Learning\Computer_Vision\Project_IntroAI\val"

for folder in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder)

    if os.path.isdir(folder_path):  # Kiểm tra nếu là folder
        all_files = os.listdir(folder_path)

        # Xóa tất cả ảnh không phải .png
        for file in all_files:
            if not file.lower().endswith(".png"):
                os.remove(os.path.join(folder_path, file))

        # Lấy lại danh sách file .png sau khi xóa
        png_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]
print("Hoàn thành!")
