import sys
from utils import combiner, processor


def main():
    args: str = "--combine"

    if args in sys.argv:
        combiner.combine_and_alpha_blend()
    else:
        processor.process()


if __name__ == "__main__":
    sys.exit(main() or 0)