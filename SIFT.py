from __future__ import print_function
import cv2
import numpy as np
np.set_printoptions(4)
import math

# Load the images
img1 = cv2.imread('D:/download/video2img/test_4/0000000010.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('D:/download/video2img/test_4/0000000016.jpg', cv2.IMREAD_GRAYSCALE)


img1 = cv2.resize(img1, (960, 540))
img2 = cv2.resize(img2, (960, 540))

sift = cv2.xfeatures2d.SIFT_create()
# Detect key points and compute descriptors
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Initialize the Brute-Force Matcher
bf = cv2.BFMatcher()

# Match descriptors
matches = bf.knnMatch(descriptors1, descriptors2, k=2)
# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

good_matches = sorted(good_matches, key = lambda x : x.distance)

point_num = 200
        
# Draw matches
matching_result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches[:point_num], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# 計算單應性矩陣 H
pts1, pts2 = [], []
for f in good_matches[:point_num]:
    pts1.append(keypoints1[f.queryIdx].pt)
    pts2.append(keypoints2[f.trainIdx].pt)
H, _ = cv2.findHomography(np.float32(pts1), np.float32(pts2), cv2.RHO)
# 計算圖像旋轉角度
angle = math.atan2(H[1,0], H[0,0])*180 /math.pi
S = H.copy()
print('\nSIFT\n單應性矩陣：')
print(H)
print('angle: ', angle)


# Show the matching result
cv2.imshow('圖1', img1)
cv2.imshow('圖2', img2)
cv2.imshow('Feature Matching Result_SIFT', matching_result)
cv2.imwrite('SIFT_result.jpg', matching_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

 
surf = cv2.xfeatures2d.SURF_create()
keypoints1, descriptors1 = surf.detectAndCompute(img1, None)
keypoints2, descriptors2 = surf.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)
# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

good_matches = sorted(good_matches, key = lambda x : x.distance)
point_num = 200
        
# Draw matches
matching_result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches[:point_num], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# 計算單應性矩陣 H
pts1, pts2 = [], []
for f in good_matches[:point_num]:
    pts1.append(keypoints1[f.queryIdx].pt)
    pts2.append(keypoints2[f.trainIdx].pt)
H, _ = cv2.findHomography(np.float32(pts1), np.float32(pts2), cv2.RHO)
# 計算圖像旋轉角度
angle = math.atan2(H[1,0], H[0,0])*180 /math.pi
S = H.copy()
print('\nSURF\n單應性矩陣：')
print(H)
print('angle: ', angle)


cv2.imshow('圖1', img1)
cv2.imshow('圖2', img2)
cv2.imshow('Feature Matching Result_SURF', matching_result)
cv2.imwrite('SURF_result.jpg', matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

