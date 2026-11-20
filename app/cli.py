import argparse
from .generate import Helper


def main():
    parser = argparse.ArgumentParser(description="Complete an anatomy study phrase.")
    parser.add_argument("phrase", nargs="*", help="Example: The right ventricle")
    args = parser.parse_args()
    helper = Helper()
    if args.phrase:
        try:
            print(helper.complete(" ".join(args.phrase)))
        except ValueError as error:
            parser.error(str(error))
        return
    print("Anatomy practice. Enter a phrase, or press Enter to quit.")
    while True:
        try:
            phrase = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not phrase:
            break
        print(helper.complete(phrase))


if __name__ == "__main__":
    main()
