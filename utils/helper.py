from colorama import Fore, Back, Style

# Helper functions for colored printing
def print_green(text):
    print(f"{Fore.GREEN}{text}{Style.RESET_ALL}", flush=True)

def print_red(text):
    print(f"{Fore.RED}{text}{Style.RESET_ALL}", flush=True)

def print_yellow(text):
    print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}", flush=True)