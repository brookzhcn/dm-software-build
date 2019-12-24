from django.core.management.base import BaseCommand
from software_mining.models import *


class Command(BaseCommand):
    help = "to predict data"

    def handle(self, *args, **options):
        TrainData.train_lr()
        TrainData.train_multiple()
