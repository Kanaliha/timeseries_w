import unittest
import subprocess
import energy_model

class TestMain(unittest.TestCase):

    def run_main(self, args):
        result = subprocess.run(['python', 'energy_model.py'] + args, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    
    def test_main_with_valid_arguments(self):
        stdout, stderr, returncode = self.run_main(['--input', 'TestInput', '--quantity', 'TestQuantity'])
        self.assertEqual(returncode, 0)
        self.assertEqual(stderr, "")

    def test_main_without_arguments(self):
        stdout, stderr, returncode = self.run_main([])
        self.assertNotEqual(returncode, 0)
        self.assertIn("the following arguments are required: --input, --quantity", stderr)


if __name__ == "__main__":
    unittest.main()