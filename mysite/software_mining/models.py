from django.db import models
import uuid
from django.db.models import Sum, Count
from sklearn import tree, svm
import numpy as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# Create your models here.


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
    filename = models.CharField(max_length=150)
    # status: modified:1066519, added:180298, renamed:76310, removed:75394
    status = models.CharField(max_length=20)
    additions = models.PositiveSmallIntegerField()
    deletions = models.PositiveSmallIntegerField()
    patch = models.TextField()
    status_encoder = None
    # give a score from Build -> Commit ->File
    # build failed: 4 build success: -1
    score = models.FloatField(blank=True, null=True, default=0)

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
    # the following field extract from commit dict
    author_name = models.CharField(max_length=50)
    author_date = models.DateTimeField()

    committer_name = models.CharField(max_length=50)
    committer_date = models.DateTimeField()

    commit_message = models.TextField()

    tree_sha = models.CharField(max_length=100)
    comment_count = models.PositiveSmallIntegerField()
    # end commit info

    parents = models.ManyToManyField('self', related_name='children', symmetrical=False)
    # status info
    additions = models.PositiveSmallIntegerField()
    deletions = models.PositiveSmallIntegerField()

    files = models.ManyToManyField(File)
    date_created = models.DateTimeField(auto_now_add=True)

    # give a score from Build -> Commit
    # build failed: 4 build success: -1
    score = models.FloatField(blank=True, null=True, default=0)

    def __str__(self):
        return "%s %s" % (self.sha, self.committer_name)

    def get_last_commit(self):
        """
        获取上次Commit 信息
        """
        try:
            return Commit.objects.filter(date_created__lt=self.date_created,
                                         project_name=self.project_name).order_by('-date_created')[0]
        except IndexError:
            return None


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
    # errored, failed, passed, errored could be ignored
    build_result = models.CharField(max_length=20)
    # relate name is build_set
    commits = models.ManyToManyField(Commit)
    date_created = models.DateTimeField(auto_now_add=True)
    objects = BuildManager()
    classifier = None

    FAILED_SCORE = 4
    PASSED_SCORE = -1

    class Meta:
        unique_together = ['project_name', 'build_id']

    def __str__(self):
        return '%s %s %s' % (self.project_name, self.build_id, self.build_result)

    def alloc_score(self):
        commits = self.commits.all()
        score = self.PASSED_SCORE if self.build_result == 'passed' else self.FAILED_SCORE
        file_num_list = []
        for commit in commits:
            file_num = commit.files.count()
            file_num_list.append(file_num)

        total = sum(file_num_list)
        for file_num, commit in zip(file_num_list, commits):
            commit.score += score * file_num/total
            commit.save(update_fields=['score'])

    def get_last_build(self):
        """
        获取上次build 信息
        """
        try:
            return Build.objects.filter(date_created__lt=self.date_created,
                                        project_name=self.project_name).order_by('-date_created')[0]
        except IndexError:
            return None

    @property
    def commit_num(self):
        return self.commits.count()

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
        failed_rate = failed_builds.count()/builds.count()
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
        print("Total: ", total, "failed: ", failed_num, "failed rate: ", failed_num/total)
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
            features = pd.sqrt([item.commit_num, item.total_deletions, item.total_additions])
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
            features = pd.sqrt([item.commit_num, item.total_deletions, item.total_additions])
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
        print(error_num/10000)

