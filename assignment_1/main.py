import cv2

image = cv2.imread('lena-1.png')
cap = cv2.VideoCapture(0)
def print_image_information(image):

    print("Image Information:")
    shape = image.shape
    print("Height: ", shape[0], " Length: ", shape[1], " channels: ", shape[2])
    size = image.size
    print("Size (number of values in the cubed array): ", size)
    d_type = image.dtype
    print("Data type: ", d_type)

def web_cam_information(cap):
    print("Web Cam Information:")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    with open("camera_outputs.txt", "w") as f:
        f.write(f"fps: {fps}\n")
        f.write(f"width: {width}\n")
        f.write(f"height: {height}")

    # open and read the file after the overwriting:
    with open("camera_outputs.txt") as f:
        print(f.read())

    cap.release()


def main():
    print_image_information(image)
    web_cam_information(cap)

if __name__ == "__main__":
    main()

