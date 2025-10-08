import cv2
import numpy as np
from matplotlib import pyplot as plt

reference_image = cv2.imread('reference_img.png')
image_to_align = cv2.imread('align_this.jpg')

good_match_precent = 0.85
max_features = 10

def harris(reference_image):
    image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    image_gray = np.float32(image_gray)
    dst = cv2.cornerHarris(image_gray, 2, 3, .04)
    dst = cv2.dilate(dst, None)
    reference_image[dst>0.01*dst.max()] = [0, 0, 255]
    cv2.imwrite(r'images\img-harris.png', reference_image)

def sift(image_to_align, reference_image, max_features, good_match_percent):
    # Convert to grayscale
    img1 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector with max_features
    sift = cv2.SIFT_create(nfeatures=max_features)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < good_match_percent * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        print("Enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    #cv2.imwrite(r'images\inlier.png', img3)
    #plt.imshow(img3, 'gray'), plt.show()

def main():
    harris(reference_image)
    sift(image_to_align, reference_image, max_features, good_match_precent)


if __name__ == "__main__":
    main()