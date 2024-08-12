import os

def checkpath(path, do_func=None, *args):
    try:
        if not os.path.exists(path):
            print(Color.WARN + f'{path} is not exist.' + Color.RESET)
            if do_func:
                print(Color.OK + f'Created {path} successfully!' + Color.RESET)
                do_func(*args)
            return False
        return True
    except PermissionError:
        print(Color.FAIL + 'Permission deny.' + Color.RESET)


def check_file(file):
    if os.path.exists(file):
        print(Color.OK + f'Created {file} successfully!' + Color.RESET)
    else:
        print(Color.FAIL + f'Failed to create {file}.' + Color.RESET)


def check_directories(*directories):
    for directory in directories:
        checkpath(directory, os.makedirs, directory)

class Color:
    FAIL = '\033[31m'  # RED
    OK = '\033[32m'  # GREEN
    WARN = '\033[33m'  # YELLOW
    INFO = '\033[34m'  # BLUE
    NOTE = '\033[35m'  # PURPLE
    MSG = '\033[36m'  # CYAN
    RED = '\033[41m'
    GREEN = '\033[42m'
    YELLOW = '\033[43m'
    BLUE = '\033[44m'
    PURPLE = '\033[45m'
    CYAN = '\033[46m'
    H_FAIL = '\033[91m'  # RED
    H_OK = '\033[92m'  # GREEN
    H_WARN = '\033[93m'  # YELLOW
    H_INFO = '\033[94m'  # BLUE
    H_NOTE = '\033[95m'  # PURPLE
    H_MSG = '\033[96m'  # CYAN
    RESET = '\033[0m'  # RESET COLOR
