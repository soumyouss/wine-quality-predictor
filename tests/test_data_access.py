from unittest import TestCase
import pandas as pd
import string
import random
from wqp.data_access import build_train_test_sets


class DataAccessTests(TestCase):

    def test_build_train_tests_sets(self):
        num_data = list(range(10))
        str_data = list(string.ascii_lowercase)[:10]
        label_data = [0]*5 + [1]*5

        for d in [num_data, str_data, label_data]:
            random.shuffle(d)

        label_col = 'label'
        df = pd.DataFrame.from_dict({
            'num_col': num_data,
            'str_col': str_data,
            label_col: label_data
        })

        train_size = 0.8
        train_test_sets = build_train_test_sets(data=df, label_col=label_col, train_size=train_size)
        expected_keys = ['train', 'test']
        self.assertSetEqual(set(expected_keys), set(train_test_sets.keys()))
        expected_train_size = int(train_size * df.shape[0])
        train_x, train_y = train_test_sets['train']
        self.assertEqual(expected_train_size, train_x.shape[0])
