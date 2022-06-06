import argparse


_PARSER = None


def get_parser():
    global _PARSER
    if _PARSER is None:
        _PARSER = argparse.ArgumentParser()
    return _PARSER


def add_argument(*args, **kwargs):
    parser = get_parser()
    parser.add_argument(*args, **kwargs)


def add_argument_group(*args, **kwargs):
    parser = get_parser()
    parser.add_argument_group(*args, **kwargs)


def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    return args
