import cv2
import numpy as np

# read image
image = cv2.imread('lena-2.png')

# Padding
border_width = 100

# Crop
shape = image.shape
image_width = shape[0]
image_height = shape[1]
x_0 = 80
x_1 = 80
y_0 = 130
y_1 = 130

# Resize
width = 20
height = 20

# Manual Copy
emptyPictureArray = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# hue
hue = 50

# Smoothing
k_size = 15

def padding(image, border_width):
    padded_image = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)
    cv2.imwrite(r'solutions\lena-2-padding.png', padded_image)


def crop(image, x_0, x_1, y_0, y_1, image_width, image_height):
    cropped_image = image[x_0:image_width - x_1, y_0:image_height - y_1]
    cv2.imwrite(r'solutions/lena-2-crop.png', cropped_image)

def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite(r'solutions/lena-2-resize.png', resized_image)


def copy(image, emptyPictureArray):
    emptyPictureArray[0:image_height, 0:image_width] = image
    cv2.imwrite(r'solutions/lena-2-copy.png', emptyPictureArray)

def grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(r'solutions/lena-2-grayscale.png', grayscale_image)

def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite(r'solutions/lena-2-hsv.png', hsv_image)

def hue_shifting(image, emptyPictureArray, hue):
    shifting = image.astype(np.uint16) + hue
    emptyPictureArray[:] = shifting.astype(np.uint8)
    cv2.imwrite(r'solutions/lena-2-hue_shifting.png', emptyPictureArray)

def smoothing(image, k_size):
    smooth_image = cv2.GaussianBlur(image, (k_size, k_size), cv2.BORDER_DEFAULT)
    cv2.imwrite(r'solutions/lena-2-smoothing.png', smooth_image)

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotate_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotate_image = cv2.rotate(image, cv2.ROTATE_180)
    cv2.imwrite(r'solutions/lena-2-rotate.png', rotate_image)
def main():
    padding(image, border_width)
    crop(image, x_1, x_0, y_1, y_0, image_width, image_height)
    resize(image, width, height)
    copy(image, emptyPictureArray)
    grayscale(image)
    hsv(image)
    hue_shifting(image, emptyPictureArray, hue)
    smoothing(image, k_size)
    rotation_angle = input("Rotate the image 90 or 180: ")
    rotation_angle = int(rotation_angle)
    rotation(image, rotation_angle)


if __name__ == "__main__":
    main()



