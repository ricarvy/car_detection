import cv2
import os
import numpy as np
from test import tester

def extract_hog(samples, logger):
    train = []
    logger.info('Extracting HOG Descriptors...')
    num = 0.
    total = len(samples)
    for f in samples:
        num += 1.
        logger.info('Processing {} {:2.1f}%'.format(f, num/total*100))
        hog = cv2.HOGDescriptor((64,128), (16,16), (8,8), (8,8), 9)
        # hog = cv2.HOGDescriptor()
        img = cv2.imread(f, -1)
        img = cv2.resize(img, (64,128))
        descriptors = hog.compute(img)
        logger.info('hog feature descriptor size: {}'.format(descriptors.shape))    # (3780, 1)
        train.append(descriptors)

    train = np.float32(train)
    train = np.resize(train, (total, 3780))

    return train

def get_svm_detector(svm):
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)

def train_svm(train, labels, logger):
    logger.info('Configuring SVM classifier.')
    svm = cv2.ml.SVM_create()
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
    svm.setC(0.01)  # From paper, soft classifier
    svm.setType(cv2.ml.SVM_EPS_SVR)

    logger.info('Starting training svm.')
    svm.train(train, cv2.ml.ROW_SAMPLE, labels)
    logger.info('Training done.')

    pwd = os.getcwd()
    model_path = os.path.join(pwd, 'svm.xml')
    svm.save(model_path)
    logger.info('Trained SVM classifier is saved as: {}'.format(model_path))

    return get_svm_detector(svm)

def hog_detect(test, svm_detector, logger):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    pwd = os.getcwd()
    test_dir = os.path.join(pwd, 'TestData')
    cv2.namedWindow('Detect')
    # for f in test:
    #     file_path = os.path.join(test_dir, f)
    #     logger.info('Processing {}'.format(file_path))
    #     img = cv2.imread(file_path)
    #     rects, _ = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8), scale=1.05)
    #     for (x,y,w,h) in rects:
    #         cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    #     # cv2.imshow('Detect', img)
    #     # c = cv2.waitKey(0) & 0xff
    #     # if c == 27:
    #     #     break
    cv2.destroyAllWindows()