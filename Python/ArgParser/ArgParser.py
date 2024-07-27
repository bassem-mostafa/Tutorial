import argparse

if __name__ == "__main__":
    print(f"Argument Parser Demo")
    parser = argparse.ArgumentParser(
                    prog="Argument Parser Demo",
                    description='Argument Parser Demo Description',
                    epilog='Argument Parser Demo Footer')
    parser.add_argument('filename')           # positional argument
    parser.add_argument('-c', '--count')      # option that takes a value
    parser.add_argument('-v', '--verbose',
                        action='store_true')  # on/off flag
    args = parser.parse_args()
    
    print(f"filename: {args.filename}")
    print(f"count: {args.count}")
    print(f"verbose: {args.verbose}")