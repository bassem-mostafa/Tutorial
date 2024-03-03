import cv2

def resize(image, size):
    """
    Resize an `image` to the required `size` maintaining the image aspect ratio with padding
    """
    size = type("size", (), dict(height=size[0], width=size[1]))
    image = type("image", (), dict(canvas=image))
    image.height, image.width = image.canvas.shape[:2]
    # Resizing on higher dimension
    scale = type("scale", (), dict(height=image.height / size.height, width = image.width / size.width))
    image.resized = type("image.resized", (), dict(height=int(image.height/max([scale.width, scale.height])), width=int(image.width/max([scale.width, scale.height]))))
    image.resized.canvas = cv2.resize(image.canvas, (image.resized.width, image.resized.height), interpolation=cv2.INTER_LINEAR)
    
    # Padding
    pad = type("pad", (), {})
    pad.left   = (size.width - image.resized.width) // 2
    pad.right  = (size.width - image.resized.width) - pad.left
    pad.top    = (size.height - image.resized.height) // 2
    pad.bottom = (size.height - image.resized.height) - pad.top
    image.resized.padded = cv2.copyMakeBorder(image.resized.canvas,
                       top=pad.top,
                       bottom=pad.bottom,
                       left=pad.left,
                       right=pad.right,
                       borderType = cv2.BORDER_CONSTANT,
                       value=0,
                       )
    return image.resized.padded

if __name__ == "__main__":
    image = cv2.imread("sample.jpg")
    resized = resize(image, (200, 400))
    print(f"resizing ({image.shape[0]}, {image.shape[1]}) -> ({resized.shape[0]}, {resized.shape[1]})")
    cv2.imshow("original", image)
    cv2.imshow("resized", resized)
    cv2.waitKey()