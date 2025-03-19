import os
from PIL import Image

def get_max_dimensions(directory):
    max_width = 0
    max_height = 0

    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                filepath = os.path.join(dirpath, filename)
                with Image.open(filepath) as img:
                    width, height = img.size
                    if width > max_width:
                        max_width = width
                    if height > max_height:
                        max_height = height

    return max_width, max_height

# Directories to check
directories = ["datasets/d1", "datasets/d2", "datasets/d3"]

# Check each directory and print the maximum dimensions
for directory in directories:
    max_width, max_height = get_max_dimensions(directory)
    print(f"Max dimensions in {directory}: {max_width}x{max_height}")

# Script was run at 19.03.2025, which yielded:
# Max dimensions in datasets/d1: 640x640
# Max dimensions in datasets/d2: 512x256
# Max dimensions in datasets/d3: 640x485