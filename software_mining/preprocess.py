import pandas as pd
import os
import json


class PreProcessError(Exception):
    pass


class PreProcessor:
    def __init__(self, data_root_directory):
        self.data_root_directory = data_root_directory

    def get_sub_folders(self):
        return os.listdir(self.data_root_directory)

    def handle_sub_folder(self, sub_folder_name, preview = 10):
        path = os.path.join(self.data_root_directory, sub_folder_name)
        if not os.path.exists(path):
            raise PreProcessError("Path dost not exist: %s" % path)
        train_set_path = os.path.join(path, 'train_set.txt')
        print(train_set_path)
        with open(train_set_path, 'r', encoding='utf-8') as f:
            train_set = json.load(f)
            passed_train_set = list(filter(lambda build: build['build_result'] == 'passed', train_set))
            failed_train_set = list(filter(lambda build: build['build_result'] == 'failed', train_set))
            print(len(passed_train_set), len(failed_train_set), sep=',')
            if preview > 0:
                for item in train_set[:10]:
                    print(item['build_id'])
                    print(item['project_name'])
                    print(item['build_result'])
                    for commit in item['commits']:
                        for key, val in commit.items():
                            print(key, val, sep="=>")
        # test_set_path = os.path.join(path, 'test_set.txt')
        # train_set_df = pd.read_json(train_set_path)
        # print(train_set_df)


if __name__ == '__main__':
    from software_mining.settings import DATA_ROOT_DIRECTORY

    processor = PreProcessor(data_root_directory=DATA_ROOT_DIRECTORY)
    print(processor.get_sub_folders())
    processor.handle_sub_folder('abarisain_dmix')
