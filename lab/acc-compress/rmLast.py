

def remove_last_word(line):
    words = line.strip().split()
    assert len(words) > 0, "Line is empty"

    return ' '.join(words[:-1])

def rmLastWords(textfile, sep):
    """
    Removes the last word from each line of a text file.
    """
    with open(textfile, 'r') as file:
        lines = file.read().split(sep)
        lines = [line.strip() for line in lines if line.strip()]

    modified_lines = [remove_last_word(line) for line in lines]
    
    return modified_lines

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("textfile", type=str)
    parser.add_argument("extractedfile", type=str)
    parser.add_argument("--sep", type=str, default="\n")

    args = parser.parse_args()

    extracted = rmLastWords(args.textfile, sep=args.sep)
    outputs = "\n".join(extracted)

    with open(args.extractedfile, 'w+') as file:
        file.write(outputs)
