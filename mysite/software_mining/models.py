from django.db import models
import uuid
from django.db.models import Sum, Count
from sklearn import tree, svm
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from django.db.models import Q
from sklearn.model_selection import train_test_split
import datetime
from django.contrib.postgres.fields import JSONField, ArrayField
# Create your models here.


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
                                         project_name=self.project_name).order_by('-committer_date')[0]
        except IndexError:
            return

    @property
    def committer_time_elapse(self):
        last_commit = self.get_last_commit()
        if last_commit:
            return (self.committer_date - last_commit.committer_date).days
        return 0

    def get_features(self):
        src_files = self.files.filter(
            Q(filename__endswith='.java') |
            Q(filename__endswith='.rb')
        )
        config_files = self.files.filter(
            Q(filename__endswith='.yml') |
            Q(filename__endswith='.xml') |
            Q(filename__endswith='.conf') |
            Q(filename__endswith='.properties') |
            Q(filename__endswith='.json') |
            Q(filename__endswith='.yaml') |
            Q(filename__endswith='.config')

        )
        src_files_info = src_files.aggregate(total_deletions=Sum('deletions'), total_additions=Sum('deletions'))
        config_files_info = config_files.aggregate(total_deletions=Sum('deletions'), total_additions=Sum('deletions'))
        return [self.committer_time_elapse,
                        src_files_info['total_deletions'] or 0, src_files_info['total_additions'] or 0,
                        src_files.count(),
                        config_files_info['total_deletions'] or 0, config_files_info['total_additions'] or 0,
                        config_files.count(),
                        ]


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
    objects = BuildManager()
    classifier = None

    class Meta:
        unique_together = ['project_name', 'build_id']

    def __str__(self):
        return '%s %s %s' % (self.project_name, self.build_id, self.build_result)

    def get_fail_rate_recently(self, limit=5):
        all_build = Build.objects.filter(project_name=self.project_name,
                                         date_created__lt=self.date_created).order_by('-date_created')[:limit]
        all_build_num = len(all_build)
        if all_build_num == 0:
            return 0
        fail_build = filter(lambda b: b.build_result == 'failed', all_build)
        return len(list(fail_build)) / all_build_num

    # def alloc_score(self):
    #     update_objects = []
    #     commits = self.commits.all()
    #     score = self.PASSED_SCORE if self.build_result == 'passed' else self.FAILED_SCORE
    #     file_num_list = []
    #     for commit in commits:
    #         file_num = commit.files.count()
    #         file_num_list.append(file_num)
    #
    #     total = sum(file_num_list)
    #     for file_num, commit in zip(file_num_list, commits):
    #         commit.score += score * file_num / total
    #         update_objects.append(commit)
    #     Commit.objects.bulk_update(update_objects, ['score'])

    @classmethod
    def alloc_scores(cls, update_objects=None):
        if update_objects is None:
            update_objects = cls.objects.all()

        for b in update_objects:
            b.alloc_score()

    @classmethod
    def alloc_score_for_specific_project(cls, project_name):
        update_objects = cls.objects.filter(project_name=project_name)
        print("allocate score for project name %s, total builds: %s" % (project_name, update_objects.count()))
        cls.alloc_scores(update_objects=update_objects)

    def get_last_build(self):
        """
        获取上次build 信息
        """
        try:
            return Build.objects.filter(date_created__lt=self.date_created,
                                        project_name=self.project_name).order_by('-date_created')[0]
        except IndexError:
            return None

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
    def train_lr(cls, limit=1000):
        # https://www.jianshu.com/p/e51e92a01a9c
        # C=1.0 : C为正则化系数λ的倒数，必须为正数，默认为1。和SVM中的C一样，值越小，代表正则化越强。
        from sklearn.linear_model import LogisticRegression
        # 先关注只有一次build的模型
        x = []
        y = []
        print("Start to prepare date ...", datetime.datetime.now())
        for build in cls.objects.filter(commit_num=1)[:limit]:
            c = build.commits.get()
            x.append([build.fail_rate_recently, *list(c.get_features())])
            label = 1 if build.build_result == 'passed' else 0
            y.append(label)
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        sc = StandardScaler().fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        lr = LogisticRegression(C=1000.0, random_state=0)
        print("Start to fit...", datetime.datetime.now())
        lr.fit(X_train_std, Y_train)
        print("start to predict", datetime.datetime.now())
        pred_test = lr.predict_proba(X_test_std)
        acc = lr.score(X_test_std, Y_test)
        print('score: %s' % acc, datetime.datetime.now())
        print(pred_test, Y_test)
        cls.classifier = lr

