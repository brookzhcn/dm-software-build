import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from .models import File, Commit, Build
import uuid

COMPILE_UNRELATED_FILE_TYPES = [
    'eot',
    'map',
    'tdd',
    'cmd',

    'jpg',
    'gitignore',
    'markdown',
    'ttf',
    'ftl',
    # style file
    'scss',
    'css',
    'yaml',
    'html',

]
COMPILE_RELATED_FILE_TYPES = [
    # config file
    'yml',
    'rb',
    'java',
    'py',
    'cmd',
    # 'xml',
    'js',
    'clj',
]


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
        self.file_name_postfix_set = set()
        self.file_name_postfix_dir = dict()
        self.build_ids = []

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
    def get_filename_postfix(filename):
        return filename.split(sep='.')[-1]

    @staticmethod
    def get_top_folder_name(filename):
        return filename.split(sep='/')[0]

    def merged_object_feature_extraction(self, merged_object: dict):
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
                # print(file)
                total_additions += file['additions']
                total_deletions += file['deletions']
                filename = file['filename']
                file_name_postfix = self.get_filename_postfix(filename)
                top_folder_name = self.get_top_folder_name(filename)
                top_folder_names.add(top_folder_name)
                if file_name_postfix in COMPILE_RELATED_FILE_TYPES:
                    if file['status'] == 'modified':
                        print("Modify: ", file['filename'])
                        total_file_modified += 1
                    elif file['status'] == 'added':
                        print("Add: ", file['filename'])
                        total_file_added += 1
                    elif file['status'] == 'deleted':
                        print("Delete: ", file['filename'])
                        total_file_deleted += 1
        has_file_add = total_file_added > 0
        has_file_deleted = total_file_deleted > 0
        features = [total_additions,
                    total_deletions,
                    has_file_add, has_file_deleted,
                    # total_file_added, total_file_deleted, total_file_modified,
                    len(file_name_postfixes), len(top_folder_names)]
        print('Feature vector: ', features)
        self.file_name_postfix_set = self.file_name_postfix_set.union(file_name_postfixes)
        return features[:4]

    def view_build_failed_info(self, sub_folder_name, filename):
        path = os.path.join(self.data_root_directory, sub_folder_name, filename)
        with open(path, 'r', encoding='utf-8') as f:
            train_set = json.load(f)
            failed_train_set = list(filter(lambda build: build['build_result'] == 'failed', train_set))
            for item in failed_train_set:
                print("\n Build id: %s" % (item['build_id']))
                for commit in item['commits']:
                    for file in commit['files']:
                        file_name_postfix = self.get_filename_postfix(file['filename'])
                        self.file_name_postfix_set.add(file_name_postfix)
                        print('File name:{} additions: {} deletions {} status {}'.format(
                            file_name_postfix, file['additions'], file['deletions'], file['status']))

    def extract_project_info(self, sub_folder_name, filename):
        path = os.path.join(self.data_root_directory, sub_folder_name, filename)
        with open(path, 'r', encoding='utf-8') as f:
            train_set = json.load(f)
            with open('/Users/zhangheng/PycharmProjects/dm-software-build/detail.json', 'w') as g:
                json.dump(train_set[1], g)
                exit(-1)

            for item in train_set:
                self.build_ids.append(int(item['build_id']))

    def view_one_build(self, sub_folder_name, filename, build_id):
        path = os.path.join(self.data_root_directory, sub_folder_name, filename)
        with open(path, 'r', encoding='utf-8') as f:
            train_set = json.load(f)
            build_item = list(filter(lambda build: build['build_id'] == build_id, train_set))[0]
            # pre_build_item = list(filter(lambda build: build['build_id'] == str(int(build_id)-1), train_set))[0]
            print(build_item)
            # print(pre_build_item)

    def get_parents(self):
        pass

    # def write_relation_to_db(self, sub_folder_name, file_name):
    #     path = os.path.join(self.data_root_directory, sub_folder_name, file_name)
    #     if not os.path.exists(path):
    #         raise PreProcessError("Path dost not exist: %s" % path)
    #     with open(path, 'r', encoding='utf-8') as f:
    #         train_set = json.load(f)
    #         for item in train_set:
    #             build_obj = Build.objects.create(
    #                 project_name=item['project_name'],
    #                 build_id=item['build_id'],
    #             )
    #             commits = item['commits']
    #             for commit in commits:
    #                 commit_obj = Commit.objects.get(
    #                     sha=commit['sha']
    #                 )
    #                 commit_info = commit['commit']
    #                 status_info = commit['stats']
    #                 assert commit_obj.additions == status_info['additions']
    #                 assert commit_obj.deletions == status_info['deletions']
    #                 assert commit_obj.commit_message == commit_info['message']
    #                 # add build -> commit relation
    #                 build_obj.commits.add(commit_obj)
    #                 parents = commit['parents']
    #                 # add parent relation
    #                 for parent in parents:
    #                     try:
    #                         parent_commit = Commit.objects.get(sha=parent['sha'])
    #                     except Commit.DoesNotExist:
    #                         print('parent commit not exits: %s' % parent['sha'])
    #                     else:
    #                         commit_obj.parents.add(parent_commit)
    #
    #                 file_ids = commit_obj.file_ids.split(',')
    #                 files = File.objects.filter(id__in=file_ids)
    #                 assert len(file_ids) == files.count()
    #                 for file in files:
    #                     # add commit -> file relation
    #                     commit_obj.files.add(file)
    #
    def write_data_to_db(self, sub_folder_name, file_name):
        path = os.path.join(self.data_root_directory, sub_folder_name, file_name)
        if not os.path.exists(path):
            raise PreProcessError("Path dost not exist: %s" % path)
        with open(path, 'r', encoding='utf-8') as f:
            # build_obj_list = []
            # commit_obj_list = []
            # file_obj_list = []
            train_set = json.load(f)
            for item in train_set:
                build_obj = Build.objects.create(
                    project_name=item['project_name'],
                    build_id=item['build_id'],
                    build_result=item['build_result'],
                )
                # build_obj_list.append(build_obj)
                commits = item['commits']
                for commit in commits:
                    if commit is None:
                        continue
                    commit_info = commit['commit']
                    status_info = commit['stats']
                    try:
                        commit_obj = Commit.objects.get(sha=commit['sha'])
                    except Commit.DoesNotExist:
                        commit_obj = Commit.objects.create(
                            sha=commit['sha'],
                            author_name=commit_info['author']['name'],
                            author_date=commit_info['author']['date'],
                            committer_name=commit_info['committer']['name'],
                            committer_date=commit_info['committer']['date'],
                            commit_message=commit_info['message'],
                            tree_sha=commit_info['tree']['sha'],
                            comment_count=commit_info['comment_count'],
                            additions=status_info['additions'],
                            deletions=status_info['deletions'],

                        )
                    build_obj.commits.add(commit_obj)
                    # commit_obj_list.append(commit_obj)
                    files = commit['files']
                    # file_ids = []
                    for file in files:
                        file_obj = File.objects.create(
                            sha=file['sha'] or '',
                            filename=file['filename'],
                            status=file['status'],
                            additions=file['additions'],
                            deletions=file['deletions'],
                            patch=file.get('patch', '')
                        )
                        commit_obj.files.add(file_obj)

            # Build.objects.bulk_create(build_obj_list)
            # Commit.objects.bulk_create(commit_obj_list, ignore_conflicts=True)
            # File.objects.bulk_create(file_obj_list, ignore_conflicts=True)
            # assert len(train_set) == Build.objects.filter(project_name=sub_folder_name).count()

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
            total_train_num = 0
            total_passed_num = 0
            total_failed_num = 0
            for item in filtered_train_set:
                print("\nbuild id: ", repr(item['build_id']))
                merged_object = self.merge_commits(item['commits'])
                # 从commit中抽取特征
                features = self.merged_object_feature_extraction(merged_object)
                y.append(item['build_result'])
                # features.append(last_build_result)
                last_build_result = item['build_result'] == 'passed'
                x.append(features)
                total_train_num += 1
                if item['build_result'] == 'passed':
                    total_passed_num += 1
                elif item['build_result'] == 'failed':
                    total_failed_num += 1

            print("\ntotal train number is: ", total_train_num)
            print("total passed number is: ", total_failed_num)
            print("total failed number is: ", total_passed_num)

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
                    if item['build_result'] == 'passed':
                        x1.append(x)
                        y1.append(y)
                    elif item['build_result'] == 'failed':
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
    DATA_ROOT_DIRECTORY = '../../../ADM2019_Mining_Challenge_Dataset'

    processor = PreProcessor(data_root_directory=DATA_ROOT_DIRECTORY)
    # processor.view_build_failed_info('apache_storm', 'train_set.txt')
    # processor.extract_project_info('apache_storm', 'train_set.txt')
    # sorted_build_ids = sorted(processor.build_ids)

    # print("\n", processor.build_ids)
    # print("\n", sorted_build_ids)
