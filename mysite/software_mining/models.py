from django.db import models
import uuid
from django.db.models import Sum, Count, Max
from sklearn import tree, svm, metrics
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from django.db.models import Q
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime
import os
import json
from sklearn import preprocessing
from django.contrib.postgres.fields import JSONField, ArrayField
# Create your models here.
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from django.conf import settings


def check_file_type(filename):
    config_types = [
        '.yml',
        '.xml',
        '.properties',
        '.json',
        '.config',
        '.yaml',
        '.conf'
    ]
    src_types = [
        '.java',
        '.rb',
    ]
    if not filename:
        return 'others'
    for c in src_types:
        if filename.endswith(c):
            return 'src'
    for c in config_types:
        if filename.endswith(c):
            return 'conf'
    return 'others'


def get_features_from_file(files):
    src_total_deletions = 0
    src_total_additions = 0
    src_file_num = 0

    conf_total_deletions = 0
    conf_total_additions = 0
    conf_file_num = 0

    src_patch_len = 0
    conf_patch_len = 0

    files_status = set()

    if files:
        for f in files:
            file_type = check_file_type(f['filename'])
            if file_type == 'src':
                src_total_additions += f.get('additions', 0)
                src_total_deletions += f.get('deletions', 0)
                src_file_num += 1
                src_patch_len += len(f.get('patch', ''))
                files_status.add(f['status'])
            elif file_type == 'conf':
                conf_total_additions += f.get('additions', 0)
                conf_total_deletions += f.get('deletions', 0)
                conf_file_num += 1
                conf_patch_len += len(f.get('patch', ''))
                files_status.add(f['status'])

    return dict(
        feature_1=src_total_deletions,
        feature_2=src_total_additions,
        feature_3=src_file_num,
        feature_4=conf_total_deletions,
        feature_5=conf_total_additions,
        feature_6=conf_file_num,
        feature_7=src_patch_len,
        feature_8=conf_patch_len,
        feature_9=len(files_status)
    )


class Committer(models.Model):
    name = models.CharField(max_length=50)
    email = models.CharField(max_length=100)
    fail_rate_overall = models.FloatField(blank=True, null=True)
    fail_rate_recently = models.FloatField(blank=True, null=True)

    class Meta:
        unique_together = ('name', 'email')


class Project(models.Model):
    name = models.CharField(max_length=50, unique=True)
    fail_rate_overall = models.FloatField(blank=True, null=True)
    data_sync = models.BooleanField(default=False)

    def set_fail_rate_overall(self):
        all_build = Build.objects.filter(project_name=self.name)
        fail_build = all_build.filter(build_result='failed')
        self.fail_rate_overall = fail_build.count() / all_build.count()
        self.save(update_fields=['fail_rate_overall'])

    #
    #     def set_fail_rate_recently(self, limit=10):
    #         all_build = Build.objects.filter(project_name=self.name)[:limit]
    #         fail_build = filter(lambda b: b.build_result == 'failed', all_build)
    #         self.fail_rate_recently = len(list(fail_build)) / len(all_build)
    #         self.save(update_fields=['fail_rate_recently'])
    #
    @classmethod
    def update_fail_rate(cls):
        for p in cls.objects.all():
            print("update fail rate for %s" % p.name)
            p.set_fail_rate_overall()
            # p.set_fail_rate_recently()

    def import_data(self, file_name='train_set.txt'):
        name = self.name
        print("\nImport data for: name ", name, datetime.datetime.now())
        if self.data_sync:
            print('Data is already imported.')
            return

        path = os.path.join(settings.DATA_ROOT_DIRECTORY, name, file_name)
        assert os.path.exists(path), "path not exist %s" % path
        num = TrainData.objects.filter(project_name=name).delete()
        print("clear data: ", num)
        with open(path, 'r', encoding='utf-8') as f:
            train_set = json.load(f)
            train_data = []
            # 之前连续passed的次数
            sequence_passed_num = 0
            # 之前连续build失败的次数
            sequence_failed_num = 0
            # valid_train_set = list(filter(lambda x: x['build_result'] in ['passed', 'failed'], train_set))
            last_build_result = None
            commit_sha_set = set()
            for build_order, build in enumerate(train_set):
                build_result = build['build_result']
                if build_result not in ['passed', 'failed']:
                    continue
                # filter None
                commits = list(filter(lambda x: x, build['commits']))
                commit_num = len(commits)

                for commit_order, commit in enumerate(commits):
                    commit_sha = commit['sha']
                    if commit_sha not in commit_sha_set:
                        commit_sha_set.add(commit_sha)
                        # feature 1 - 10 is for file
                        files = commit['files']
                        file_features = get_features_from_file(files=files)
                        commit_info = commit['commit']
                        obj = TrainData(
                            commit_sha=commit_sha,
                            committer_name=commit_info['committer']['name'],
                            commit_order=commit_order,
                            comment_count=commit_info['comment_count'],
                            commit_message=commit_info['message'],
                            project_name=name,
                            build_id=build['build_id'],
                            build_order=build_order,
                            commit_num=commit_num,
                            build_result=build['build_result'],
                            last_build_result=last_build_result,
                            feature_11=len(commit['parents']),
                            feature_12=sequence_failed_num,
                            feature_13=sequence_passed_num,
                            **file_features
                        )
                        train_data.append(obj)
                # update last build result at end
                last_build_result = build_result

                if build_result == 'passed':
                    sequence_passed_num += 1
                    sequence_failed_num = 0
                else:
                    sequence_failed_num += 1
                    sequence_passed_num = 0
            print("prepare to update %s" % len(train_data))
            TrainData.objects.bulk_create(train_data, batch_size=1000)
            self.data_sync = True
            self.save()


class File(models.Model):
    """
    sha只是一个文件的唯一识别码，但是一次改动中sha并不唯一，比如rename操作和revert操作导致文件sha一样，
    因而不能用sha做主键
    src file: *.java *.rb
    config file: *.xml *.yml
    """
    # for query
    project_name = models.CharField(max_length=50, blank=True, null=True)
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    sha = models.CharField(max_length=100)
    filename = models.TextField()
    # status: modified:1066519, added:180298, renamed:76310, removed:75394
    status = models.CharField(max_length=20)
    additions = models.PositiveSmallIntegerField()
    deletions = models.PositiveSmallIntegerField()
    patch = models.TextField()
    status_encoder = None
    # give a score from Build -> Commit ->File
    # build failed: 4 build success: -1
    score = models.FloatField(blank=True, null=True, default=0)

    class Meta:
        abstract = True

    def __str__(self):
        return "%s del: %s add: %s status:%s sha: %s" % (self.filename,
                                                         self.deletions,
                                                         self.additions,
                                                         self.status,
                                                         self.sha)

    @property
    def filename_postfix(self):
        return self.filename.split(sep='.')[-1]

    @classmethod
    def set_status_encoder(cls):
        # File.objects.values('status').annotate(models.Count('id'))
        le = LabelEncoder()
        le.fit(['added', 'status', 'removed', 'renamed'])
        cls.status_encoder = le

    @property
    def is_src_type(self):
        src_types = [
            '.java',
            '.rb',
        ]
        for c in src_types:
            if self.filename.endswith(c):
                return True
        return False

    @property
    def is_config_type(self):
        config_types = [
            '.yml',
            '.xml',
            '.properties',
            '.json',
            '.config',
            '.yaml',
            '.conf'
        ]
        for c in config_types:
            if self.filename.endswith(c):
                return True
        return False


class Commit(models.Model):
    sha = models.CharField(max_length=100, primary_key=True, db_index=True)
    # for query
    project_name = models.CharField(max_length=50, blank=True, null=True)
    commit_order = models.PositiveIntegerField()

    # the following field extract from commit dict
    author = JSONField(null=True)
    committer = JSONField(null=True)
    # for search
    committer_name = models.CharField(max_length=100)
    committer_date = models.DateTimeField()
    commit_message = models.TextField()
    tree = JSONField(null=True)
    comment_count = models.IntegerField()
    # end commit info
    parents = JSONField(null=True)
    # status info
    additions = models.IntegerField()
    deletions = models.IntegerField()
    files = JSONField(null=True)
    compute_features = ArrayField(models.FloatField(null=True), null=True)

    # give a score from Build -> Commit
    # build failed: 4 build success: -1

    def __str__(self):
        return "%s %s" % (self.sha, self.committer)

    def get_last_commit(self):
        """
        获取上次Commit 信息
        """
        try:
            return Commit.objects.filter(committer_date__lt=self.committer_date,
                                         committer_name=self.committer_name,
                                         project_name=self.project_name).order_by('-commit_order')[0]
        except IndexError:
            return

    @property
    def committer_time_elapse(self):
        last_commit = self.get_last_commit()
        if last_commit:
            return (self.committer_date - last_commit.committer_date).days
        return 0

    def get_features(self):
        src_total_deletions = 0
        src_total_additions = 0
        src_file_num = 0

        conf_total_deletions = 0
        conf_total_additions = 0
        conf_file_num = 0

        patch_len = 0
        committer_time_elapse = self.committer_time_elapse
        files = self.files

        if files:
            for f in files:
                file_type = f['file_type']
                if file_type == 'src':
                    src_total_additions += f['additions']
                    src_total_deletions += f['deletions ']
                    src_file_num += 1
                    patch_len += f['patch_len']
                elif file_type == 'conf':
                    conf_total_additions += f['additions']
                    conf_total_deletions += f['deletions ']
                    conf_file_num += 1
                    patch_len += f['patch_len']

        features = [
            src_total_deletions,
            src_total_additions,
            src_file_num,
            conf_total_deletions,
            conf_total_additions,
            conf_file_num,
            patch_len,
            committer_time_elapse
        ]
        # if date_set:
        #     self.compute_features = features
        #     self.save(update_fields=['compute_features'])
        return features

    @classmethod
    def update_features(cls, project_name):
        print('updating project: %s' % project_name, datetime.datetime.now())
        update_objects = []
        objs = cls.objects.filter(compute_features=None, project_name=project_name)
        num = 0
        for obj in objs:
            num += 1
            obj.compute_features = obj.get_features()
            update_objects.append(obj)
        print('total update: ', num)
        Commit.objects.bulk_update(update_objects, fields=['compute_features'], batch_size=1000)


class BuildManager(models.Manager):
    """
    ignore all errored object
    if commit is None, then see the last build or output passed
    """

    def get_queryset(self):
        return super().get_queryset().exclude(build_result='errored').exclude(commits=None)


class Build(models.Model):
    project_name = models.CharField(max_length=50)
    build_id = models.CharField(max_length=20)
    build_order = models.IntegerField()
    # errored, failed, passed, errored could be ignored
    build_result = models.CharField(max_length=20)
    commit_num = models.PositiveIntegerField(default=0)
    fail_rate_recently = models.FloatField(null=True, blank=True)
    # relate name is build_set
    commits = ArrayField(models.TextField(), blank=True)
    # objects = BuildManager()
    classifier = None

    class Meta:
        unique_together = ['project_name', 'build_id']

    def __str__(self):
        return '%s %s %s' % (self.project_name, self.build_id, self.build_result)

    def get_last_build(self):
        try:
            return Build.objects.get(build_order=self.build_order - 1, project_name=self.project_name)
        except Build.DoesNotExist:
            return

    def get_fail_rate_recently(self, limit=5):
        build_order = self.build_order
        all_build = Build.objects.filter(project_name=self.project_name,
                                         build_order__lt=build_order).order_by('-build_order')[:limit]
        all_build_num = len(all_build)
        if all_build_num == 0:
            return 0
        fail_build = filter(lambda b: b.build_result == 'failed', all_build)
        return len(list(fail_build)) / all_build_num

    @classmethod
    def update_commit_num_and_fail_rate(cls):
        for obj in cls.objects.all():
            print(obj.pk)
            obj.commit_num = obj.commits.count()
            obj.fail_rate_recently = obj.get_fail_rate_recently()
            obj.save(update_fields=['commit_num', 'fail_rate_recently'])

    @property
    def total_additions(self):
        return self.commits.aggregate(total_additions=Sum('additions'))['total_additions']

    @property
    def total_deletions(self):
        return self.commits.aggregate(total_additions=Sum('deletions'))['total_additions']

    @classmethod
    def get_all_projects(cls):
        """
        return a queryset {'project_name': 'AChep_AcDisplay', 'build_num': 201}
        """
        return cls.objects.values('project_name').annotate(build_num=models.Count('build_id'))

    @classmethod
    def describe_project(cls, project_name):
        builds = cls.objects.filter(project_name=project_name)
        failed_builds = builds.filter(build_result='failed')
        failed_rate = failed_builds.count() / builds.count()
        print('\n', project_name)
        print("failed rate: ", failed_rate)

    @classmethod
    def describe_projects(cls):
        """
        some project has no build failed samples
        failed : passed = 1:4
        predicate rate should > 80%
        """
        total = cls.objects.count()
        failed_num = cls.objects.filter(build_result='failed').count()
        print("Total: ", total, "failed: ", failed_num, "failed rate: ", failed_num / total)
        projects = cls.get_all_projects()
        for project in projects:
            project_name = project['project_name']
            cls.describe_project(project_name)

    @classmethod
    def train(cls):
        x = []
        y = []
        for item in cls.objects.all()[:10000]:
            assert isinstance(item, Build)
            features = np.sqrt([item.commit_num, item.total_deletions, item.total_additions])
            x.append(
                [
                    # item.project_name,
                    *features
                ]
            )
            y.append(item.build_result)
        clf = tree.DecisionTreeClassifier(max_depth=15)
        clf.fit(x, y)

        cls.classifier = clf

    @classmethod
    def test(cls):
        x = []
        y = []
        error_num = 0
        for item in cls.objects.all()[10000:20000]:
            assert isinstance(item, Build)
            features = np.sqrt([item.commit_num, item.total_deletions, item.total_additions])
            x.append(
                [
                    # item.project_name,
                    *features
                ]
            )
            y.append(item.build_result)

        predict_y = cls.classifier.predict(x)
        for y1, y2 in zip(y, predict_y):
            if y1 != y2:
                error_num += 1
        print(error_num / 10000)

    @classmethod
    def get_success_rate_by_commit_num(cls, commit_num):
        total = cls.objects.filter(commit_num=commit_num)
        if total.count() > 10:
            return total.filter(build_result='passed').count() / total.count()
        return

    @classmethod
    def plot_success_rate_vs_commit_num(cls):
        x = []
        y = []
        m = Build.objects.aggregate(max=Max('commit_num'))['max']
        for num in range(1, m + 1):
            success_rate = cls.get_success_rate_by_commit_num(num)
            if success_rate is not None:
                print(num, success_rate)
                x.append(num)
                y.append(success_rate)
        return x, y

    @classmethod
    def train_lr(cls, project_name=None, limit=3000):
        first_builds = Build.objects.filter(build_order=0)
        first_build_success = first_builds.filter(build_result='passed').count() / first_builds.count()

        # https://www.jianshu.com/p/e51e92a01a9c
        # C=1.0 : C为正则化系数λ的倒数，必须为正数，默认为1。和SVM中的C一样，值越小，代表正则化越强。
        from sklearn.linear_model import LogisticRegression
        # 先关注只有一次build的模型
        x = []
        y = []
        if project_name is not None:
            total_builds = Build.objects.filter(project_name=project_name, commit_num=1)
        else:
            total_builds = cls.objects.filter(commit_num=1)[:limit]
        print("Start to prepare data ...", datetime.datetime.now())
        fail_rate_dict = {}
        for p in Project.objects.all():
            fail_rate_dict[p.name] = p.fail_rate_overall

        for build in total_builds:
            c = Commit.objects.get(sha=build.commits[0])
            last_build = build.get_last_build()
            if last_build is None:
                last_build_success = first_build_success
            elif last_build.build_result == 'passed':
                last_build_success = 1
            else:
                last_build_success = 0
            features = c.compute_features if c.compute_features else list(c.get_features())
            features = np.log(np.add(features, [1] * len(features)))
            p = fail_rate_dict[build.project_name]
            x.append([last_build_success, p, *features])
            # label = 1 if build.build_result == 'passed' else 0
            y.append(build.build_result)
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        X_train_std = X_train
        X_test_std = X_test
        # sc = MinMaxScaler(feature_range=(0, 1))
        # X_train_std = sc.fit_transform(X_train)
        # X_test_std = sc.fit_transform(X_test)
        lr = LogisticRegression(C=10.0, random_state=0)
        # ros = RandomOverSampler(random_state=0)
        # X_resampled, y_resampled = ros.fit_sample(X_train_std, Y_train)
        # rus = RandomUnderSampler(random_state=0)

        # X_resampled, y_resampled = rus.fit_sample(X_train_std, Y_train)
        print("Start to fit...", datetime.datetime.now())
        lr.fit(X_train_std, Y_train)
        # lr.fit(X_resampled, y_resampled)
        print("start to predict", datetime.datetime.now())
        pred_test_prob = lr.predict_proba(X_test_std)
        pred_test = lr.predict(X_test_std)

        acc = lr.score(X_test_std, Y_test)
        report = metrics.classification_report(Y_test, pred_test)
        print('score: %s' % acc, datetime.datetime.now())
        print(pred_test_prob, Y_test, '\n', report)
        cls.classifier = lr


class TrainData(models.Model):
    commit_sha = models.CharField(max_length=100, primary_key=True, db_index=True)
    committer_name = models.CharField(max_length=100)
    commit_order = models.PositiveIntegerField(default=0)
    comment_count = models.PositiveIntegerField(default=0)
    commit_message = models.TextField()
    # build info
    project_name = models.CharField(max_length=50)
    build_id = models.CharField(max_length=20)
    build_order = models.IntegerField()
    commit_num = models.PositiveIntegerField(default=0)
    # errored, failed, passed, errored could be ignored
    build_result = models.CharField(max_length=20)
    # 只关注上一次build结果，如果是首次可以用首次平均build成功率或者直接去掉
    last_build_result = models.CharField(max_length=20, null=True)
    # features used
    feature_1 = models.FloatField(null=True, blank=True)
    feature_2 = models.FloatField(null=True, blank=True)
    feature_3 = models.FloatField(null=True, blank=True)
    feature_4 = models.FloatField(null=True, blank=True)
    feature_5 = models.FloatField(null=True, blank=True)
    feature_6 = models.FloatField(null=True, blank=True)
    feature_7 = models.FloatField(null=True, blank=True)
    feature_8 = models.FloatField(null=True, blank=True)
    feature_9 = models.FloatField(null=True, blank=True)
    feature_10 = models.FloatField(null=True, blank=True)
    feature_11 = models.FloatField(null=True, blank=True)
    feature_12 = models.FloatField(null=True, blank=True)
    feature_13 = models.FloatField(null=True, blank=True)
    feature_14 = models.FloatField(null=True, blank=True)
    feature_15 = models.FloatField(null=True, blank=True)
    test = models.BooleanField(default=False)
    # logistical regression output for commit more than one
    predict_result = models.FloatField(null=True, blank=True)

    classifier = None

    @classmethod
    def train_lr(cls, project_names=None, limit=None, commit_num=1):
        # first_builds = Build.objects.filter(build_order=0)
        # first_build_success = first_builds.filter(build_result='passed').count() / first_builds.count()

        # https://www.jianshu.com/p/e51e92a01a9c
        # C=1.0 : C为正则化系数λ的倒数，必须为正数，默认为1。和SVM中的C一样，值越小，代表正则化越强。
        from sklearn.linear_model import LogisticRegression
        # 先关注只有一次build的模型
        X = []
        Y = []
        if project_names is not None:
            total_commits = cls.objects.filter(project_name__in=project_names, commit_num=commit_num).exclude(
                last_build_result=None)
        else:
            total_commits = cls.objects.filter(commit_num=commit_num).exclude(last_build_result=None)
        print("Start to prepare data ...", datetime.datetime.now())
        fail_rate_dict = {}
        for p in Project.objects.all():
            fail_rate_dict[p.name] = p.fail_rate_overall
        total_commits = total_commits.values_list(
            'build_result',
            'last_build_result',
            'comment_count',
            'feature_1',
            'feature_2',
            'feature_3',
            'feature_4',
            'feature_5',
            'feature_6',
            'feature_7',
            'feature_8',
            'feature_9',
            'feature_11',
            'feature_12',
            'feature_13',
        )
        if limit is not None:
            total_commits = total_commits[:limit]
        for item in total_commits:
            label, last_build, *features = item
            last_build = 1 if last_build == 'passed' else 0
            features = np.log(np.add(features, [1] * len(features)))
            Y.append(label)
            X.append([last_build, *features])
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
        X_train_std = X_train
        X_test_std = X_test
        # sc = MinMaxScaler(feature_range=(0, 1))
        # X_train_std = sc.fit_transform(X_train)
        # X_test_std = sc.fit_transform(X_test)
        lr = LogisticRegression(C=10.0, random_state=0)
        # ros = RandomOverSampler(random_state=0)
        # X_resampled, y_resampled = ros.fit_sample(X_train_std, Y_train)
        # rus = RandomUnderSampler(random_state=0)

        # X_resampled, y_resampled = rus.fit_sample(X_train_std, Y_train)
        print("Start to fit...", datetime.datetime.now())
        lr.fit(X_train_std, Y_train)
        # lr.fit(X_resampled, y_resampled)
        print("start to predict", datetime.datetime.now())
        pred_test_prob = lr.predict_proba(X_test_std)
        pred_test = lr.predict(X_test_std)

        acc = lr.score(X_test_std, Y_test)
        report = metrics.classification_report(Y_test, pred_test)
        print('score: %s' % acc, datetime.datetime.now())
        print(pred_test_prob, Y_test, '\n', report)
        cls.classifier = lr
