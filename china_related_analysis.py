# coding:utf-8
import jieba
import json
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
from sklearn.svm import SVC
import glob
import pickle

from sklearn.model_selection import cross_val_score
# from sklearn.cross_validation import cross_val_score


class TestSklearn(object):
    @classmethod
    def from_class(cls):
        attrs = ['person_names', 'place_names', 'firms', 'cultures', 'others']
        files = [u'人名', u'地点', u'机构', u'文娱', u'代指中国or中国人']
        if not hasattr(cls, 'dicts'):
            cls.dicts = []
            for k, v in zip(attrs, files):
                cls.dicts.append(cls.get_words(v))
        return cls()

    @staticmethod
    def get_words(filepath):
        # 遍历指定目录，显示目录下的所有文件名，并生成向量字典
        word_set = set()
        for filename in glob.glob('%s/*' % filepath):
            with open(filename, 'r') as f:
                for item in f.readlines():
                    item = item.strip().replace('《', '').replace('》', '')
                    if item not in word_set:
                        jieba.add_word(item)
                        word_set.add(unicode(item))
        return word_set

    def create_vector(self, item):
        feature_vector = [0] * 15

        def scan_vector(_word, _index):
            for vector_dict in self.dicts:
                if _word in vector_dict:
                    feature_vector[_index] += 1
                _index += 1

        indexs = {u'标题': 0,
                  u'摘要': 5,
                  u'内容': 10}
        for key, index in indexs.items():
            seg_list = jieba.cut(item[key], cut_all=False)
            [scan_vector(word, index) for word in seg_list]
        return feature_vector


    def getXY(self, filename):
        #feature_vectors作为x输入,tags作为y
        feature_vectors = []
        tags = []
        with open(filename) as src:
            items = json.load(src, encoding='utf-8')
            for item in items:
                tag = {u'Y': 1, u'N': 0}[item.pop(u'标签')]
                tags.append(tag)
                feature_vector = self.create_vector(item)
                feature_vectors.append(feature_vector)
        return feature_vectors, tags


if __name__ == '__main__':
    # 构建SVM分类器
    # test_sklearn = TestSklearn.from_class()
    # try:
    #     with open('SVC.pickle') as f:
    #         clf = pickle.load(f)
    # except:
    #     filenames = ['backup/tag%d.json' % i for i in [1, 2, 5]]
    #     feature_vectors = []
    #     tags = []
    #     for filename in filenames:
    #         _feature_vectors, _tags = test_sklearn.getXY(filename)
    #         feature_vectors += _feature_vectors
    #         tags += _tags
    #     X = np.array(feature_vectors)
    #     y = np.array(tags)
    #     clf = SVC()
    #     clf.fit(X, y)
    #     with open('SVC.pickle', 'w') as f:
    #         pickle.dump(clf, f)
    # feature_vectors = []
    # tags = []
    # filenames = ['backup/tag%d.json' % i for i in [7, 8, 9]]
    # for filename in filenames:
    #     _feature_vectors, _tags = test_sklearn.getXY(filename)
    #     feature_vectors += _feature_vectors
    #     tags += _tags
    # # 机器分类结果results
    # wrong_result = []
    # results = (clf.predict(feature_vectors))
    #
    # count = 0
    # count_fenzi = 0
    # count_fenmu = 0
    # for i in range(len(tags)):
    #     if tags[i] == results[i]:
    #         count += 1
    #     # else:
    #     #     wrong_result.append(i)
    #     if tags[i] == 1:
    #         count_fenmu += 1
    #         if results[i] == 1:
    #             count_fenzi += 1
    # print float(count) / float(len(tags))
    # print float(count_fenzi) / float(count_fenmu)
    #十则交叉检验
    test_sklearn = TestSklearn.from_class()
    feature_vectors = []
    tags = []
    filenames = ['backup/tag%s.json' % i for i in [1, 2, 5, 7, 8, 9]]
    # for num in range(1, 7):
    #     filenames.append('backup/tag%s.json' % num)
    # filenames.append('backup/tag6.json')
    for filename in filenames:
        _features, _tags = test_sklearn.getXY(filename)
        feature_vectors.extend(_features)
        tags.extend(_tags)
    X = np.array(feature_vectors)
    y = np.array(tags)
    clf = SVC()
    scores = cross_val_score(clf, X, y, cv=10)
    print scores
    print scores.mean()