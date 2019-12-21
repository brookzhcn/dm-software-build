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
        projects = sorted(self.processor.get_sub_folders())
        # self.processor.write_data_to_db('sensu_sensu-community-plugins', 'train_set.txt')
        # exit(-1)
        print("Total projects: %s" % len(projects))
        index = 370
        untest_projects = projects[index:]
        for project_name in untest_projects:
            index += 1
            print('\nProject %s: %s' % (index, project_name))
            if project_name.startswith('.') or project_name.endswith('.csv') or project_name.endswith('.txt'):
                continue
            try:
                self.processor.write_data_to_db(project_name, 'train_set.txt')
            except Exception as e:
                print(str(e))
                print(index, project_name)
                exit(-1)
