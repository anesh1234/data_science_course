from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import config

def add_whitespace(img, bboxes:pd.DataFrame, h, w):
    """
    :param img: original image
    :param bboxes: bboxes as pandas dataframe where each row is 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
    :param h: resized height dimension of image
    :param w: resized width dimension of image
    :return: dictionary containing {image:new_image, bboxes:['x_min', 'y_min', 'x_max', 'y_max', "class_id"]}
    """

    # Create a new image with the desired size and white background
    white_img = Image.new("RGB", (w, h), (255, 255, 255))

    # Calculate the position to paste the original image onto the new image
    paste_x = (w - img.width) // 2
    paste_y = (h - img.height) // 2

    # Paste the original image onto the new image
    white_img.paste(img, (paste_x, paste_y))

    # Adjust bounding boxes accordingly
    adjusted_bboxes = bboxes.copy()
    adjusted_bboxes['xmin'] += paste_x
    adjusted_bboxes['ymin'] += paste_y
    adjusted_bboxes['xmax'] += paste_x
    adjusted_bboxes['ymax'] += paste_y

    return {"image": white_img, "bboxes": adjusted_bboxes}

test_img = Image.open("datasets/radiography_scaled/test/0004_jpg.rf.6434bfce7667ea786e5f251dc0d8b8b1.jpg")
filename = '0004_jpg.rf.6434bfce7667ea786e5f251dc0d8b8b1.jpg'

# Get bounding boxes and labels for the current image
anns = pd.read_csv('datasets/radiography_scaled/test/_annotations.csv')

# Convert the 'class' column to index values and reduce the dataframe
anns['class'] = anns['class'].map(config.class_to_index)
bbox_labels = anns[anns['filename'] == filename][['xmin', 'ymin', 'xmax', 'ymax', 'class']]

# Perform the resizing
result = add_whitespace(test_img, bbox_labels, 640, 640)

# # Extract the resized image array and bounding boxes
# new_img = result["image"]
# new_bbox_labels = result["bboxes"]

# # Show the results
# new_img.show()
# new_img.save(f'tools/testing/RESIZED_{filename}')
# print(new_bbox_labels)



# Plotting the results versus the original
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Plot the new image WITHOUT bounding boxes on the top left
axes[0,0].imshow(result["image"])
axes[0,0].set_title("Resized Image Without Boxes")
axes[0,0].axis("off")

# Plot the new image WITH bounding boxes on the top right
axes[0,1].imshow(result["image"])
for _, bbox in result["bboxes"].iterrows():
    x_min, y_min, x_max, y_max, class_id = bbox
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         linewidth=2, edgecolor='r', facecolor='none')
    axes[0,1].add_patch(rect)
    axes[0,1].text(x_min, y_min - 10, f"{config.index_to_class[class_id]}", color='r', fontsize=12)
axes[0,1].set_title("Resized Image with Boxes")
axes[0,1].axis("off")

#########################################################################################

# Plot the original image WITHOUT bounding boxes on the bottom left
original_img = Image.open("tools/testing/0004_jpg.rf.6434bfce7667ea786e5f251dc0d8b8b1.jpg")

axes[1,0].imshow(original_img)
axes[1,0].set_title("Original Image Without Boxes")
axes[1,0].axis("off")

# Plot the original image WITH bounding boxes on the bottom right
axes[1,1].imshow(original_img)
for _, bbox in bbox_labels.iterrows():
    x_min, y_min, x_max, y_max, class_id = bbox
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         linewidth=2, edgecolor='r', facecolor='none')
    axes[1,1].add_patch(rect)
    axes[1,1].text(x_min, y_min - 10, f"{config.index_to_class[class_id]}", color='r', fontsize=12)
axes[1,1].set_title("Original Image with Boxes")
axes[1,1].axis("off")

plt.show()
