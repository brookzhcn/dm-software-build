from django.core.management.base import BaseCommand
from software_mining.models import *
from sklearn.externals import joblib
import pickle

class Command(BaseCommand):
    help = "to predict data"

    def handle(self, *args, **options):
        project_names = Project.objects.filter(fail_rate_overall__lt=0.9, fail_rate_overall__gt=0.1).values_list('name')
        project_names = [p[0] for p in project_names]
        # TrainData.train_lr(project_names)

        # TrainData.classifier.save()
        # TrainData.train_multiple()
        joblib.dump(TrainData.classifier, 'simple_classifier.pkl')
        joblib.dump(TrainData.multi_classifier, 'multiple_classifier.pkl')
        # for p in Project.objects.all():
        #     TrainData.predict(p.name)
        # TrainData.train_multiple()
        clf = joblib.load('simple_classifier.pkl')
        TrainData.classifier = clf
        TrainData.train_multiple()

