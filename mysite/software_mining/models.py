from django.db import models
import uuid

# Create your models here.


class File(models.Model):
    """
    sha只是一个文件的唯一识别码，但是一次改动中sha并不唯一，比如rename操作和revert操作导致文件sha一样，
    因而不能用sha做主键
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    sha = models.CharField(max_length=100)
    filename = models.CharField(max_length=150)
    # status: modified, added, renamed, deleted
    status = models.CharField(max_length=20)
    additions = models.PositiveSmallIntegerField()
    deletions = models.PositiveSmallIntegerField()
    patch = models.TextField()

    def __str__(self):
        return "%s del: %s add: %s status:%s sha: %s" % (self.filename,
                                                         self.deletions,
                                                         self.additions,
                                                         self.status,
                                                         self.sha)


class Commit(models.Model):
    sha = models.CharField(max_length=100, primary_key=True, db_index=True)
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
    # help fix relations
    file_ids = models.TextField(blank=True)
    date_created = models.DateTimeField(auto_now_add=True)


class Build(models.Model):
    project_name = models.CharField(max_length=50)
    build_id = models.CharField(max_length=20)
    build_result = models.CharField(max_length=20)
    commits = models.ManyToManyField(Commit)
    date_created = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['project_name', 'build_id']
