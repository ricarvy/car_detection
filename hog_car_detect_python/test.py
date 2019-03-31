# coding:utf-8

import cv2
import numpy as np

def tester(file):
    camera = cv2.VideoCapture(file)
    kernel = np.ones((5, 5), np.uint8)
    background = None

    while camera.isOpened():
        # 读入摄像头的帧
        ret, frame = camera.read()
        # 把第一帧作为背景
        if ret == True:
            if background is None:
                background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                background = cv2.GaussianBlur(background, (21, 21), 0)
                continue
            # 读入帧
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 高斯平滑 模糊处理 减小光照 震动等原因产生的噪声影响
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
            diff = cv2.absdiff(background, gray_frame)
            # 将区别转为二值
            diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            # 定义结构元素
            es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
            diff = cv2.dilate(diff, es, iterations=2)
            image, cnts, hierarcchy = cv2.findContours(diff.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < 1500:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("contours", frame)
            cv2.imshow("diff", diff)
        else:
            camera.release()
            cv2.waitKey(0)
        if cv2.waitKey(5) & 0xff == ord("q"):
            break

    cv2.destroyAllWindows()
    camera.release()