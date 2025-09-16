import cv2
import numpy as np

image = cv2.imread('lambo.png')
image_shapes = cv2.imread('shapes-1.png')

threshold_1 = 50
threshold_2 = 50

template = cv2.imread('shapes_template.jpg')

scale_factor = 2
up_or_down = "up"

def sobel_edge_detection(image):
   img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   img_blur = cv2.GaussianBlur(img_grey, (3,3), 0)

    # The picture becomes very "washed out" when I do it like this:
   image_sobel = cv2.Sobel(img_blur, cv2.CV_64F, 1, 1, 1)
   cv2.imwrite(f"images/lambo_sobel.png", image_sobel)

    # If I make a sobel edge detection for x-axes, then y-axes and use magnitude to combine them,
    # the image becomes much clearer. I did not find any better solution to strenghten the egde at each pixel.
    # Then converting to uint8 to scale into the valid 0-255 scale to not include negative pixel values
    # https://opencv.org/blog/edge-detection-using-opencv/
   sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1,0,1)
   sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0,1,1)
   image_sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
   image_sobel_magnitude = cv2.convertScaleAbs(image_sobel_magnitude)
   cv2.imwrite(f"images/lambo_sobel_magnitude.png", image_sobel_magnitude)

def canny_edge_detecetion(image, threshold_1, threshold_2):
    image_blur = cv2.GaussianBlur(image, (3,3), 0)
    image_canny = cv2.Canny(image_blur, threshold_1, threshold_2)
    cv2.imwrite(f"images/lambo_canny.png", image_canny)

def template_match(image_shapes, template):
    image_gray = cv2.cvtColor(image_shapes, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    w, h = template_gray.shape[::-1]

    res = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(image_shapes, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    cv2.imwrite("images/shapes_template_matching.png", image_shapes)


def resize(image, scale_factor, up_or_down):
    if up_or_down == "up":
        for _ in range(scale_factor):
            image = cv2.pyrUp(image)
    elif up_or_down == "down":
        for _ in range(scale_factor):
            image = cv2.pyrDown(image)

    cv2.imwrite("images/resized_image.png", image)



def main():
    sobel_edge_detection(image)
    canny_edge_detecetion(image, threshold_1, threshold_2)
    template_match(image_shapes, template)
    resize(image, scale_factor, up_or_down)


if __name__ == "__main__":
    main()