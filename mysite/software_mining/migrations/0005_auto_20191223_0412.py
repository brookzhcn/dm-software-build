# Generated by Django 3.0.1 on 2019-12-22 20:12

import django.contrib.postgres.fields
import django.contrib.postgres.fields.jsonb
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('software_mining', '0004_auto_20191223_0353'),
    ]

    operations = [
        migrations.AddField(
            model_name='commit',
            name='compute_features',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(null=True), null=True, size=None),
        ),
        migrations.AlterField(
            model_name='commit',
            name='author',
            field=django.contrib.postgres.fields.jsonb.JSONField(null=True),
        ),
        migrations.AlterField(
            model_name='commit',
            name='committer',
            field=django.contrib.postgres.fields.jsonb.JSONField(null=True),
        ),
        migrations.AlterField(
            model_name='commit',
            name='files',
            field=django.contrib.postgres.fields.jsonb.JSONField(null=True),
        ),
        migrations.AlterField(
            model_name='commit',
            name='parents',
            field=django.contrib.postgres.fields.jsonb.JSONField(null=True),
        ),
        migrations.AlterField(
            model_name='commit',
            name='tree',
            field=django.contrib.postgres.fields.jsonb.JSONField(null=True),
        ),
    ]