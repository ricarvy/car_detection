import numpy as np
import os

class Dataset:
    def __init__(self):
        pass

    def load_data_set(self, logger):
        logger.info('Checking data path!')
        pwd = os.getcwd()
        logger.info('Current path is:{}'.format(pwd))

        # 提取正样本
        pos_dir = os.path.join(pwd, 'Positive')
        if os.path.exists(pos_dir):
            logger.info('Positive data path is:{}'.format(pos_dir))
            pos = os.listdir(pos_dir)
            logger.info('Positive samples number:{}'.format(len(pos)))

        # 提取负样本
        neg_dir = os.path.join(pwd, 'Negative')
        if os.path.exists(neg_dir):
            logger.info('Negative data path is:{}'.format(neg_dir))
            neg = os.listdir(neg_dir)
            logger.info('Negative samples number:{}'.format(len(neg)))

        # 提取测试集
        test_dir = os.path.join(pwd, 'TestData')
        if os.path.exists(test_dir):
            logger.info('Test data path is:{}'.format(test_dir))
            test = os.listdir(test_dir)
            logger.info('Test samples number:{}'.format(len(test)))

        return pos, neg, test

    def load_train_samples(self, pos, neg):
        pwd = os.getcwd()
        pos_dir = os.path.join(pwd, 'Positive')
        neg_dir = os.path.join(pwd, 'Negative')

        samples = []
        labels = []
        for f in pos:
            file_path = os.path.join(pos_dir, f)
            if os.path.exists(file_path):
                samples.append(file_path)
                labels.append(1.)

        for f in neg:
            file_path = os.path.join(neg_dir, f)
            if os.path.exists(file_path):
                samples.append(file_path)
                labels.append(-1.)

        # labels 要转换成numpy数组，类型为np.int32
        labels = np.int32(labels)
        labels_len = len(pos) + len(neg)
        labels = np.resize(labels, (labels_len, 1))

        return samples, labels