# Generated by Django 3.0.1 on 2019-12-24 09:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('software_mining', '0005_auto_20191223_0412'),
    ]

    operations = [
        migrations.CreateModel(
            name='TrainData',
            fields=[
                ('commit_sha', models.CharField(db_index=True, max_length=100, primary_key=True, serialize=False)),
                ('committer_name', models.CharField(max_length=100)),
                ('commit_order', models.PositiveIntegerField(default=0)),
                ('comment_count', models.PositiveIntegerField(default=0)),
                ('commit_message', models.TextField()),
                ('project_name', models.CharField(max_length=50)),
                ('build_id', models.CharField(max_length=20)),
                ('build_order', models.IntegerField()),
                ('commit_num', models.PositiveIntegerField(default=0)),
                ('build_result', models.CharField(max_length=20)),
                ('last_build_result', models.CharField(max_length=20, null=True)),
                ('feature_1', models.FloatField(blank=True, null=True)),
                ('feature_2', models.FloatField(blank=True, null=True)),
                ('feature_3', models.FloatField(blank=True, null=True)),
                ('feature_4', models.FloatField(blank=True, null=True)),
                ('feature_5', models.FloatField(blank=True, null=True)),
                ('feature_6', models.FloatField(blank=True, null=True)),
                ('feature_7', models.FloatField(blank=True, null=True)),
                ('feature_8', models.FloatField(blank=True, null=True)),
                ('feature_9', models.FloatField(blank=True, null=True)),
                ('feature_10', models.FloatField(blank=True, null=True)),
                ('feature_11', models.FloatField(blank=True, null=True)),
                ('feature_12', models.FloatField(blank=True, null=True)),
                ('feature_13', models.FloatField(blank=True, null=True)),
                ('feature_14', models.FloatField(blank=True, null=True)),
                ('feature_15', models.FloatField(blank=True, null=True)),
                ('test', models.BooleanField(default=False)),
                ('predict_result', models.FloatField(blank=True, null=True)),
            ],
        ),
    ]
