from sklearn import tree, svm
from sklearn.model_selection import cross_val_score
import numpy as np


class Classifier:
    def __init__(self):
        self.model = None

    def train(self, x, y):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(x, y)
        self.model = clf

    def validate(self, x, y):
        clf = svm.SVC(kernel='linear', C=1)
        # clf = tree.DecisionTreeClassifier()
        scores = cross_val_score(clf, x, y, cv=10)
        print(scores, np.mean(scores))
        return scores

    def test(self, x, y):
        assert self.model is not None
        predict_y = self.model.predict(x)
        # print(predict_y, y)

    def plot(self):
        assert self.model is not None
        tree.plot_tree(self.model)


if __name__ == '__main__':
    from software_mining.settings import DATA_ROOT_DIRECTORY
    from software_mining.preprocess import PreProcessor
    # project_name = 'abarisain_dmix'
    # project_name = 'rspec_rspec-core'
    # project_name = 'codeforamerica_adopt-a-hydrant'
    project_name = 'apache_storm'
    # project_name = 'justinfrench_formtastic'
    # project_name = 'activescaffold_active_scaffold'
    # project_name = 'eirslett_frontend-maven-plugin'
    print("Project name: ", project_name)
    processor = PreProcessor(data_root_directory=DATA_ROOT_DIRECTORY)
    train_x, train_y = processor.get_x_and_y(project_name, 'train_set.txt')
    classifier = Classifier()
    classifier.train(train_x, train_y)
    # test_x, test_y = processor.get_x_and_y(project_name, 'test_set.txt', lambda build: True)
    # classifier.test(test_x, test_y)

    classifier.validate(train_x, train_y)

    print('Done')


