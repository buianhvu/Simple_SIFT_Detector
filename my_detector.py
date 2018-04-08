import cv2
import numpy as np

LOWEST_MATCHES_NUMBER = 30

sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher();

train_img = cv2.imread('Photo/demo2.jpg', 0)
train_kp, train_desc = sift.detectAndCompute(train_img, None);

camera = cv2.VideoCapture(0);
while (True):
    det, frame_with_color = camera.read();
    frame = cv2.cvtColor(frame_with_color,cv2.COLOR_BGR2GRAY)
    frame_kp, frame_desc = sift.detectAndCompute(frame,None)
    matches=bf.knnMatch(frame_desc,train_desc,k=2)
    good = []
    for m,n in matches:
        if(m.distance < 0.75*n.distance):
            good.append(m)
    if(len(good)> LOWEST_MATCHES_NUMBER):
        train_points = []
        frame_points = []
        for m in good:
            train_points.append(train_kp[m.trainIdx].pt)
            frame_points.append(frame_kp[m.queryIdx].pt)
        train_points, frame_points=np.float32((train_points,frame_points))
        H,status=cv2.findHomography(train_points,frame_points,cv2.RANSAC,3.0)
        h,w=train_img.shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(frame_with_color,[np.int32(queryBorder)],True,(0,0,255),5)
    else:
        print('FOUND LOW MATCHES NUMBER {} / {}'.format(len(good), LOWEST_MATCHES_NUMBER))
    cv2.imshow('result',frame_with_color)
    if cv2.waitKey(5)==ord('q'):
        break
camera.release()
cv2.destroyAllWindows()




