import unittest

from jl_opt_preflight import validate_token_split


class Split:
    def __init__(self, ids=None, mask=None):
        self.rows = dict(input_ids=[ids if ids is not None else [4, 5] + [2] * 510],
                         attention_mask=[mask if mask is not None else [1, 1] + [0] * 510],
                         label=[0])

    def __len__(self):
        return 1

    def __getitem__(self, key):
        return {name: values[key] for name, values in self.rows.items()}


class CacheRowsTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(validate_token_split(Split(), 64, 2), 1)

    def test_invalid(self):
        for split in (Split(ids=[4]), Split(ids=[64] + [2] * 511),
                      Split(ids=[2] * 512), Split(mask=[0] * 512),
                      Split(mask=[1, 0, 1] + [0] * 509),
                      Split(ids=[4, 5] + [3] * 510),
                      Split(mask=[1, 3] + [0] * 510)):
            with self.subTest(split=split), self.assertRaises(ValueError):
                validate_token_split(split, 64, 2)


if __name__ == '__main__':
    unittest.main()
