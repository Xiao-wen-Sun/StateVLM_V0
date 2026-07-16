from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

# Data containing image paths and bounding boxes
data = [
    {
        "img_path": "image__00005_.png",
        "bbox": [0.39, 0.58, 0.72, 0.86]
    }
]

# Path to the image folder
image_folder = "/data/sun/20250523-TableDataset5Templates/template2/Template2/_final_images/"

def draw_bbox_on_image(img_path, bbox, color="red", width=3):
    """
    Open an image, draw a bounding box, and return the image.
    
    :param img_path: Path to the image file
    :param bbox: Bounding box in relative coordinates [x1, y1, x2, y2]
    :param color: Color of the bounding box
    :param width: Width of the bounding box line
    :return: PIL Image object with bounding box drawn
    """
    image = Image.open(img_path)
    img_width, img_height = image.size

    # Convert relative bbox to absolute pixel coordinates
    x1 = bbox[0] * img_width
    y1 = bbox[1] * img_height
    x2 = bbox[2] * img_width
    y2 = bbox[3] * img_height
    abs_bbox = [x1, y1, x2, y2]

    draw = ImageDraw.Draw(image)
    draw.rectangle(abs_bbox, outline=color, width=width)

    return image

def show_image(image, figsize=(8, 8)):
    """Display a PIL Image using matplotlib."""
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Process each image in the data
for item in data:
    img_path = os.path.join(image_folder, item["img_path"])
    bbox = item["bbox"]

    image_with_bbox = draw_bbox_on_image(img_path, bbox)
    show_image(image_with_bbox)
