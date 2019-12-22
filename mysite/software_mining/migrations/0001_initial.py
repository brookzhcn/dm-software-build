# Generated by Django 3.0.1 on 2019-12-21 12:38

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='File',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('sha', models.CharField(max_length=100)),
                ('filename', models.CharField(max_length=150)),
                ('status', models.CharField(max_length=20)),
                ('additions', models.PositiveSmallIntegerField()),
                ('deletions', models.PositiveSmallIntegerField()),
                ('patch', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='Commit',
            fields=[
                ('sha', models.CharField(db_index=True, max_length=100, primary_key=True, serialize=False)),
                ('author_name', models.CharField(max_length=50)),
                ('author_date', models.DateTimeField()),
                ('committer_name', models.CharField(max_length=50)),
                ('committer_date', models.DateTimeField()),
                ('commit_message', models.TextField()),
                ('tree_sha', models.CharField(max_length=100)),
                ('comment_count', models.PositiveSmallIntegerField()),
                ('additions', models.PositiveSmallIntegerField()),
                ('deletions', models.PositiveSmallIntegerField()),
                ('date_created', models.DateTimeField(auto_now_add=True)),
                ('files', models.ManyToManyField(to='software_mining.File')),
                ('parents', models.ManyToManyField(related_name='children', to='software_mining.Commit')),
            ],
        ),
        migrations.CreateModel(
            name='Build',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('project_name', models.CharField(max_length=50)),
                ('build_id', models.CharField(max_length=20)),
                ('build_result', models.CharField(max_length=20)),
                ('date_created', models.DateTimeField(auto_now_add=True)),
                ('commits', models.ManyToManyField(to='software_mining.Commit')),
            ],
            options={
                'unique_together': {('project_name', 'build_id')},
            },
        ),
    ]