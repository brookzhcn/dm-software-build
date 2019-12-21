from django.test import TestCase
from .preprocess import PreProcessor
from django.conf import settings

DATA_ROOT_DIRECTORY = settings.DATA_ROOT_DIRECTORY


# Create your tests here.


class PreprocessTestCase(TestCase):
    def setUp(self):
        self.processor = PreProcessor(data_root_directory=DATA_ROOT_DIRECTORY)

    def test_view_build_failed_info(self):
        self.processor.view_build_failed_info('apache_storm', 'train_set.txt')

    def test_write_data_to_db(self):
        self.processor.write_data_to_db('sensu_sensu-community-plugins', 'train_set.txt')
        # for project_name in sorted(self.processor.get_sub_folders()):
        #     try:
        #         self.processor.write_data_to_db(project_name, 'train_set.txt')
        #     except Exception as e:
        #         print(project_name)
        #         print(str(e))
        #         exit(-1)
