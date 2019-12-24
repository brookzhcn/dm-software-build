from django.core.management.base import BaseCommand
from software_mining.models import *


class Command(BaseCommand):
    help = "to predict data"

    def handle(self, *args, **options):
        TrainData.train_lr()
        for p in Project.objects.all():
            TrainData.predict(p.name)
        # TrainData.train_multiple()
