import os
from PIL import Image

def resize_images(root_dir, max_size=(224, 224)):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(dirpath, filename)
                try:
                    with Image.open(file_path) as img:
                        img.thumbnail(max_size)
                        img.save(file_path)
                        print(f"Resized {file_path}")
                except Exception as e:
                    print(f"Failed to resize {file_path}: {e}")

if __name__ == "__main__":
    resize_images("data/images/single_search")
