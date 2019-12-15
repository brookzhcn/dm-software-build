import pandas as pd
import os
import json
import matplotlib.pyplot as plt


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

    def get_x_and_y(self, project_name, file_name,
                    filter_func=lambda build: build['build_result'] in ['failed', 'passed']):
        x, y = self.handle_project(project_name, file_name, filter_func, preview=0)
        return x, y

    @staticmethod
    def merge_commits(commits):
        """
        为了方便特征提取，需要将多个commits merge在一起
        :param commits: 多次提交列表
        :return: a merged object
        """
        merged_object = {}
        for commit in commits:
            for key, val in commit.items():
                if key not in merged_object:
                    merged_object[key] = [val]
                else:
                    merged_object[key].append(val)
        return merged_object

    @staticmethod
    def merged_object_feature_extraction(merged_object: dict):
        files = merged_object['files']
        # 所有commit中文件中增加的改动地方
        total_additions = 0
        # 所有commit中文件中删除的改动地方
        total_deletions = 0

        # 所有commit中增加的文件数目
        total_file_added = 0
        # 所有commit中删除的文件数目：画图可以看出，该特征很重要
        total_file_deleted = 0
        # 所有commit中修改的文件数目
        total_file_modified = 0

        # 所有commit中所有改动文件的后缀：有可能某些文件后缀影响build成功率
        file_name_postfixes = set()
        # 所有commit中所有改动文件的顶级目录：有些目录下面的文件改动会影响build成功率
        top_folder_names = set()

        for file_list in files:
            for file in file_list:
                print(file)
                total_additions += file['additions']
                total_deletions += file['deletions']
                file_names = file['filename'].split(sep='.')
                file_name_postfixes.add(file_names[-1])
                top_folder_names.add(file_names[0].split('/')[0])
                if file['status'] == 'modified':
                    total_file_modified += 1
                elif file['status'] == 'added':
                    total_file_added += 1
                elif file['status'] == 'deleted':
                    total_file_deleted += 1
        print(total_additions, total_deletions, file_name_postfixes, top_folder_names,
              total_file_modified, total_file_added, total_file_deleted)

        return total_additions, total_deletions, total_file_added, total_file_deleted, total_file_modified, \
            len(file_name_postfixes), len(top_folder_names)

    def handle_project(self, sub_folder_name, file_name, filter_func, preview: int = 10):
        path = os.path.join(self.data_root_directory, sub_folder_name)
        if not os.path.exists(path):
            raise PreProcessError("Path dost not exist: %s" % path)
        train_set_path = os.path.join(path, file_name)
        with open(train_set_path, 'r', encoding='utf-8') as f:
            train_set = json.load(f)
            # passed_train_set = list(filter(lambda build: build['build_result'] == 'passed', train_set))
            # failed_train_set = list(filter(lambda build: build['build_result'] == 'failed', train_set))
            filtered_train_set = filter(filter_func, train_set)
            # print(len(passed_train_set), len(failed_train_set), sep=',')
            x = []
            y = []
            for item in filtered_train_set:
                merged_object = self.merge_commits(item['commits'])
                # 从commit中抽取特征
                features = self.merged_object_feature_extraction(merged_object)
                y.append(item['build_result'])
                x.append(features)

            if preview > 0:
                x1 = []
                y1 = []
                x2 = []
                y2 = []
                for item in train_set[:preview]:
                    merged_object = self.merge_commits(item['commits'])
                    features = self.merged_object_feature_extraction(merged_object)
                    if features[0] > 4000 or features[1] > 4000:
                        print(*features, item['build_result'])
                    if item['build_result'] == 'passed' and x < 500 and y < 500:
                        x1.append(x)
                        y1.append(y)
                    elif item['build_result'] == 'failed' and x < 500 and y < 500:
                        x2.append(x)
                        y2.append(y)

                fig, ax = plt.subplots()
                print(x1, y1)
                print(x2, y2)
                ax.scatter(x1, y1, c='tab:blue', label='passed', alpha=0.3, edgecolors='none')
                ax.scatter(x2, y2, c='tab:orange', label='failed', alpha=0.3, edgecolors='none')
                ax.set_xlabel('add', fontsize=15)
                ax.set_ylabel('delete', fontsize=15)
                ax.legend()
                ax.grid(True)
                plt.show()
                print('done')
            return x, y


if __name__ == '__main__':
    from software_mining.settings import DATA_ROOT_DIRECTORY

    processor = PreProcessor(data_root_directory=DATA_ROOT_DIRECTORY)
    processor.handle_project('abarisain_dmix', 'train_set.txt', preview=10)
