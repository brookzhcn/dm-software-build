from sklearn import tree


class Classifier:
    def __init__(self):
        self.model = None

    def train(self, x, y):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(x, y)
        self.model = clf

    def test(self, x, y):
        assert self.model is not None
        predict_y = self.model.predict(x)
        print(predict_y, y)

    def plot(self):
        assert self.model is not None
        tree.plot_tree(self.model)


if __name__ == '__main__':
    from software_mining.settings import DATA_ROOT_DIRECTORY
    from software_mining.preprocess import PreProcessor

    processor = PreProcessor(data_root_directory=DATA_ROOT_DIRECTORY)
    train_x, train_y = processor.get_x_and_y('abarisain_dmix', 'train_set.txt')
    classifier = Classifier()
    classifier.train(train_x, train_y)
    test_x, test_y = processor.get_x_and_y('abarisain_dmix', 'test_set.txt', lambda build: True)
    classifier.test(test_x, test_y)

    print('Done')


