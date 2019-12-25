from django.core.management.base import BaseCommand
from software_mining.models import *


class Command(BaseCommand):
    help = "to predict data"

    def handle(self, *args, **options):
        project_names = Project.objects.filter(fail_rate_overall__lt=0.9, fail_rate_overall__gt=0.1).values_list('name')
        project_names = [p[0] for p in project_names]
        TrainData.train_lr(project_names)
        # for p in Project.objects.all():
        #     TrainData.predict(p.name)
        # TrainData.train_multiple()
