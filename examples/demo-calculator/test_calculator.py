import unittest

from calculator import add, divide, subtract


class CalculatorTests(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_subtract(self):
        self.assertEqual(subtract(7, 4), 3)

    def test_divide(self):
        self.assertEqual(divide(8, 2), 4)

    def test_divide_by_zero_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "^division by zero$"):
            divide(8, 0)


if __name__ == "__main__":
    unittest.main()
