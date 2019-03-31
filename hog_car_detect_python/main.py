# *_*coding:utf-8 *_*
import sys
import logging
from test import tester

from methods import extract_hog, get_svm_detector, train_svm, hog_detect
from dataset_class import Dataset

def logger_init():
    # 获取logger实例，如果参数为空则返回root logger
    logger = logging.getLogger("PedestranDetect")

    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger

if __name__ == '__main__':
    file_path = 'data/test.mp4'
    logger = logger_init()
    dataset = Dataset()
    pos, neg, test = dataset.load_data_set(logger=logger)
    samples, labels = dataset.load_train_samples(pos, neg)
    train = extract_hog(samples, logger=logger)
    logger.info('Size of feature vectors of samples: {}'.format(train.shape))
    logger.info('Size of labels of samples: {}'.format(labels.shape))
    svm_detector = train_svm(train, labels, logger=logger)
    hog_detect(test, svm_detector, logger)
    tester(file_path)

