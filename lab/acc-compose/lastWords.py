

def extractLastWords(textfile, sep):
    """
    Extracts the last word from each line of a text file.
    """
    with open(textfile, 'r') as file:
        lines = file.read().split(sep)

    last_words = [line.strip().split()[-1] for line in lines if line.strip()]
    last_words = [word.strip() for word in last_words if word.strip()]  # Filter out empty strings
    
    return last_words   

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("textfile", type=str)
    parser.add_argument("extractedfile", type=str)
    parser.add_argument("--sep", type=str, default="\n")

    args = parser.parse_args()

    extracted = extractLastWords(args.textfile, sep=args.sep)

    outputs = "\n".join(extracted)
    with open(args.extractedfile, 'w') as file:
        file.write(outputs)