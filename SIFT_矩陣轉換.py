from __future__ import print_function
import cv2
import numpy as np
np.set_printoptions(4)
import math

# Load the images
img1 = cv2.imread('D:/download/video2img/v100_dataset/test/video_0001/image_02/data_t/0000000000.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('D:/download/video2img/v100_dataset/test/video_0001/image_02/data_t/0000000006.jpg', cv2.IMREAD_GRAYSCALE)


img1 = cv2.resize(img1, (960, 340))
img2 = cv2.resize(img2, (960, 340))

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

c = 0
total_m = 0
total_dx = 0
total_dy = 0
# for i in good_matches[:point_num]:
#     print('圖1點索引: ', i.queryIdx, '坐標: ', keypoints1[i.queryIdx].pt)
#     print('圖2點索引: ', i.trainIdx, '坐標: ', keypoints2[i.trainIdx].pt)
#     print('兩點歐氏距離: ', i.distance)

#     #dy = (img2.shape[0] - keypoints1[i.queryIdx].pt[1]) - (img2.shape[0] - keypoints2[i.trainIdx].pt[1])
#     dy = keypoints1[i.queryIdx].pt[1] - keypoints2[i.trainIdx].pt[1]
#     dx = keypoints1[i.queryIdx].pt[0] - keypoints2[i.trainIdx].pt[0]
#     total_dx += dx
#     total_dy += dy
    
#     m = dy/dx
#     total_m += m
    
#     print('x軸位移', dx, '\ny軸位移', dy)
#     print('斜率: ', m)
    
#     c = c + 1
#     print(c)
#     print('='*50)

# print('平均x軸變化量: ', total_dx/c)
# print('平均y軸變化量: ', total_dy/c)

# mean_m = total_m/c
# print('平均斜率: ', mean_m)

# 計算單應性矩陣 H
pts1, pts2 = [], []
for f in good_matches[:point_num]:
    pts1.append(keypoints1[f.queryIdx].pt)
    pts2.append(keypoints2[f.trainIdx].pt)
H, _ = cv2.findHomography(np.float32(pts1), np.float32(pts2), cv2.RHO)

# 計算圖像旋轉角度
angle = math.atan2(H[1,0], H[0,0])*180 /math.pi



S = H.copy()

print('單應性矩陣：')
print(H)

print('angle: ', angle)

# Translate
H[0] = H[0]/H[0, 0]
print('')
print('將第一列第一項X係數變為1')
print(H)

H[1] = H[1] - H[0]*H[1][0]
H[2] = H[2] - H[0]*H[2][0]
print('')
print('將第二列第一項X係數變為0')
print(H)

H[1] = H[1]/H[1][1]
H[0] = H[0] - H[1]*H[0][1]
H[2] = H[2] - H[1]*H[2][1]
print('')
print('將第二列第二項Y係數變為1，將第一列第二項Y係數變為0')
print(H)

H[2] = H[2]/H[2, 2]
print('')
print('將第三列第三項Z係數變為1')
print(H)


# Scale about origin
S[1] = S[1] - S[0]*(S[1, 0]/S[0, 0])
S[2] = S[2] - S[0]*(S[2, 0]/S[0, 0])
S = np.around(S, 6)
print('')
print('將第二、三列第一項X係數變為0')
print(S)

S[0] = S[0] - S[1]*(S[0, 1]/S[1, 1])
S[2] = S[2] - S[1]*(S[2, 1]/S[1, 1])
S = np.around(S, 6)
print('')
print('將第一、三列第二項Y係數變為0')
print(S)

S[0] = S[0] - S[2]*(S[0, 2]/S[2, 2])
S[1] = S[1] - S[2]*(S[1, 2]/S[2, 2])
S[2] = S[2]/S[2, 2]
print('')
print('將第一、二列第三項Z係數變為0，第三列第三項Z係數變為1')
print(S)



# Show the matching result
cv2.imshow('圖1', img1)
cv2.imshow('圖2', img2)
cv2.imshow('Feature Matching Result', matching_result)
cv2.imwrite('SIFT_result.jpg', matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()


