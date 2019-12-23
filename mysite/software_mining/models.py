from django.db import models
import uuid
from django.db.models import Sum, Count, Max
from sklearn import tree, svm
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from django.db.models import Q
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime
from django.contrib.postgres.fields import JSONField, ArrayField
# Create your models here.
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


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
#
#     def set_fail_rate_overall(self):
#         all_build = Build.objects.filter(project_name=self.name)
#         fail_build = all_build.filter(build_result='failed')
#         self.fail_rate_overall = fail_build.count() / all_build.count()
#         self.save(update_fields=['fail_rate_overall'])
#
#     def set_fail_rate_recently(self, limit=10):
#         all_build = Build.objects.filter(project_name=self.name)[:limit]
#         fail_build = filter(lambda b: b.build_result == 'failed', all_build)
#         self.fail_rate_recently = len(list(fail_build)) / len(all_build)
#         self.save(update_fields=['fail_rate_recently'])
#
#     @classmethod
#     def update_fail_rate(cls):
#         for p in cls.objects.all():
#             print("update fail rate for %s" % p.name)
#             p.set_fail_rate_overall()
#             p.set_fail_rate_recently()


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

    def get_features(self, date_set=False):
        if self.compute_features:
            return self.compute_features
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
        if date_set:
            self.compute_features = features
            self.save(update_fields=['compute_features'])
        return features


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
            return Build.objects.get(build_order=self.build_order-1, project_name=self.project_name)
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
        for num in range(1, m+1):
            success_rate = cls.get_success_rate_by_commit_num(num)
            if success_rate is not None:
                print(num, success_rate)
                x.append(num)
                y.append(success_rate)
        return x, y

    @classmethod
    def train_lr(cls, project_name=None, limit=3000):
        first_builds = Build.objects.filter(build_order=0)
        first_build_success = first_builds.filter(build_result='passed').count()/ first_builds.count()

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
        for build in total_builds:
            c = Commit.objects.get(sha=build.commits[0])
            last_build = build.get_last_build()
            if last_build is None:
                last_build_success = first_build_success
            elif last_build.build_result == 'passed':
                last_build_success = 1
            else:
                last_build_success = 0

            x.append([last_build_success, *list(c.get_features(date_set=True))])
            label = 1 if build.build_result == 'passed' else 0
            y.append(label)
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        sc = MinMaxScaler(feature_range=(0, 1))
        X_train_std = sc.fit_transform(X_train)
        X_test_std = sc.fit_transform(X_test)
        lr = LogisticRegression(C=1000.0, random_state=0)
        ros = RandomOverSampler(random_state=0)
        # X_resampled, y_resampled = ros.fit_sample(X_train_std, Y_train)
        # rus = RandomUnderSampler(random_state=0)

        # X_resampled, y_resampled = rus.fit_sample(X_train_std, Y_train)
        print("Start to fit...", datetime.datetime.now())
        lr.fit(X_train_std, Y_train)
        # lr.fit(X_resampled, y_resampled)
        print("start to predict", datetime.datetime.now())
        pred_test = lr.predict_proba(X_test_std)
        acc = lr.score(X_test_std, Y_test)
        print('score: %s' % acc, datetime.datetime.now())
        print(pred_test, Y_test)
        cls.classifier = lr

