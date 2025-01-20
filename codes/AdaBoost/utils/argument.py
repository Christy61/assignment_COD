from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def get_parser():
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--base_dir", type=str,
                        help="dataset dir.")
    
    parser.add_argument("--load", type=str, default=None,
                        help="load from")
    
    parser.add_argument("--gpu", type=str, default='0',
                        help="which gpu to use")

    parser.add_argument('--output', type=str,
                        help="Optionally save output as img.")

    return parser