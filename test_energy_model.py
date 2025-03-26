import unittest
import subprocess
import numpy as np
import pandas as pd
from energy_model import EnergyModel


class TestMain(unittest.TestCase):

    def run_main(self, args):
        result = subprocess.run(['python', 'energy_model.py'] + args, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    
    def test_main_with_invalid_arguments(self):
        _, _, returncode = self.run_main(['--invalidParam1', 'TestInput', '--invalidParam2', 'TestQuantity'])
        self.assertNotEqual(returncode, 0)

    def test_main_without_arguments(self):
        _, stderr, returncode = self.run_main([])
        self.assertNotEqual(returncode, 0)
        self.assertIn("the following arguments are required: --input, --quantity", stderr)

    # ...


class TestEnergyModel(unittest.TestCase):
    def setUp(self):
        test_file_path = './test_data/SG.csv'
        quantity = 'Consumption'
        self.energy_model = EnergyModel(test_file_path,quantity)

    def test_loading_data(self):
        self.energy_model.load_data()
        self.assertIsNotNone(self.energy_model.data_df)

    def test_adding_features(self):
        self.energy_model.load_data()
        self.energy_model.add_features()
        # wanted features
        self.assertIsNotNone(self.energy_model.data_df['day_of_year'])
        self.assertIsNotNone(self.energy_model.data_df['day_of_week'])
        self.assertIsNotNone(self.energy_model.data_df['is_weekend'])
        self.assertIsNotNone(self.energy_model.data_df['hours'])

        # unwanted features
        with self.assertRaises(KeyError):
            self.energy_model.data_df['not_existing']

    # ...

if __name__ == "__main__":
    unittest.main()