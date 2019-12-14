import pandas as pd
import os
import json


class PreProcessError(Exception):
    pass


class PreProcessor:
    """
    每一个项目下面对应唯一的一个project_name
    每次build对应唯一的build_id
    每次build有唯一的build_result: errored, passed, failed, 我们只关心passed和failed，相当于二分类问题
    影响build结果主要由每次的commits，而一次build可能对应多个commit

    单次commit属性有：
    1. sha： hash code 无意义的唯一识别码
    2. commit： 提供人员信息（author，committer） 相当于每次git commit时的日志信息，参考不大
    3. url：提交的查询地址，无意义
    4. html_url：网页查看提交内容的地址， 无意义
    5. comments_url: 提交备注的查看地址，无意义
    6. author： 作者相关信息，意义不大，可以参考
    7. committer： 提交者信息，同上 {'type': 'User', 'site_admin': False}
    8. parents： 上次提交，可以通过sha来找到上一次
    9. stats： total： 总共的修改地方  additions: 增加的地方个数 deletions： 删除的地方个数
    10. files: 对应9， 说明文件的改动个数
      10.1 status： modified/added/deleted
      10.2 additions： 和 9 对应
      10.3 deletions： 和 9 对应
      10.4 changes： 应该是additions+ deletions

    9， 10 应该是有用的

    """
    def __init__(self, data_root_directory):
        self.data_root_directory = data_root_directory

    def get_sub_folders(self):
        return os.listdir(self.data_root_directory)

    def handle_sub_folder(self, sub_folder_name, preview=10):
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
                for item in train_set[:preview]:
                    print(item['build_id'])
                    print(item['project_name'])
                    print(item['build_result'])
                    for commit in item['commits']:
                        for key, val in commit.items():
                            print(key, val, sep="=>")
                    print("=================================================================")
        # test_set_path = os.path.join(path, 'test_set.txt')
        # train_set_df = pd.read_json(train_set_path)
        # print(train_set_df)


if __name__ == '__main__':
    from software_mining.settings import DATA_ROOT_DIRECTORY

    processor = PreProcessor(data_root_directory=DATA_ROOT_DIRECTORY)
    processor.handle_sub_folder('abarisain_dmix', preview=10)
