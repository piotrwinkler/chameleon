import sys
sys.path.append('C:/STUDIA/INZYNIERKA/chameleon')
import os
import argparse
from loguru import logger as log


def main():
    args = parse_args()
    version = args.version
    log.debug(f"Validating version: {version}")

    TEST_PARAMETERS = f"data/configs/test_parameters_{version}.json"
    if not os.path.isfile(TEST_PARAMETERS):
        log.error(f"No file {TEST_PARAMETERS}")
        raise Exception(f"No file {TEST_PARAMETERS}")

    RETRAINING_NET_DIRECTORY = f"model_states/{version}/fcn_model{version}_epoch_final.pth"
    if not os.path.isfile(RETRAINING_NET_DIRECTORY):
        log.error(f"No file {RETRAINING_NET_DIRECTORY}")
        raise Exception(f"No file {RETRAINING_NET_DIRECTORY}")

    log.debug(f"Version {version} - OK")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', help='chosen config version',
                        required=True, type=str)

    return parser.parse_args()


if __name__ == "__main__":
    main()
