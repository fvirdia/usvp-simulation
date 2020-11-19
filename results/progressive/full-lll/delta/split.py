import argparse
import json

def main():
    parser = argparse.ArgumentParser(description='Split result files.')
    parser.add_argument('filename', type=str, help='file to be split')
    args = parser.parse_args()

    print(args.filename)

    fn = args.filename

    with open(fn) as f:
        data = json.load(f)

    for tag in data:
        out = { tag: data[tag] }
        with open(f"{fn}-{tag}.json", 'w') as f:
            json.dump(out, f)

main()
