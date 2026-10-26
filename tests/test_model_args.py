import io
import sys
import unittest
from argparse import ArgumentParser
from unittest import mock

from arguments import ModelParams, get_combined_args


class ModelArgsTest(unittest.TestCase):
    def parse(self, config):
        parser = ArgumentParser()
        ModelParams(parser, sentinel=True)
        with mock.patch.object(sys, 'argv', ['prog', '-m', 'rgb_model']), \
                mock.patch('builtins.open', side_effect=(FileNotFoundError()
                           if config is None else lambda *args, **kwargs: io.StringIO(config))):
            return get_combined_args(parser)

    def test_model_without_cfg_args_uses_cli(self):
        self.assertEqual(self.parse(None).model_path, 'rgb_model')

    def test_cfg_args_accepts_literal_namespace(self):
        args = self.parse("Namespace(source_path='scene', sh_degree=2)")
        self.assertEqual(args.source_path, 'scene')
        self.assertEqual(args.model_path, 'rgb_model')

    def test_cfg_args_does_not_execute_code(self):
        with self.assertRaises(ValueError):
            self.parse("Namespace(source_path=__import__('os').getcwd())")


if __name__ == '__main__':
    unittest.main()
