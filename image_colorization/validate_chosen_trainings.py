import sys
sys.path.append('C:/STUDIA/INZYNIERKA/chameleon')
import os
import argparse
from loguru import logger as log


def main():
    args = parse_args()
    version = args.version
    log.debug(f"Validating version: {version}")

    TRAINING_PARAMETERS = f"data/configs/training_parameters_{version}.json"
    if not os.path.isfile(TRAINING_PARAMETERS):
        log.error(f"No file {TRAINING_PARAMETERS}")
        raise Exception(f"No file {TRAINING_PARAMETERS}")
    else:
        log.debug(f"Version {version} - OK")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', help='chosen config version',
                        required=True, type=str)

    return parser.parse_args()


if __name__ == "__main__":
    main()
