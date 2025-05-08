import os

base_dir = "nominal"

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    normalized_label = folder.lower().replace(" ", "")

    files = sorted(os.listdir(folder_path))
    for i, file in enumerate(files):
        ext = os.path.splitext(file)[1]  # .jpg, .png, dll
        new_name = f"img_{normalized_label}_{str(i+1).zfill(4)}{ext}"
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)
        print(f"{old_path} → {new_path}")
