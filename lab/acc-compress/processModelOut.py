

def testLine(line, ref:str):
    refWords = ref.split()
    lineWords = line.split()
    
    nWords = len(refWords)

    for i in range(len(refWords) - 1):
        if refWords[i] != lineWords[i]:
            assert "should not happen"

    if len(lineWords) < nWords or lineWords[nWords-1] != refWords[-1]:
        return False
    else:
        return True 

def testFile(modelOutFile, refFile):
    with open(modelOutFile) as mo, open(refFile) as ref:
        modelLines  = mo.read().split(sep="#"*6)
        modelLines  = [ l.strip() for l in modelLines if l.strip() ]
        refLines    = [ l.strip() for l in ref.readlines() if l.strip() ]

        assert len(modelLines) == len(refLines), f"Number of lines in {modelOutFile} and {refFile} do not match: {len(modelLines)} vs {len(refLines)}"
        
        correctLines = 0
        for ml, rl in zip(modelLines, refLines):
            if testLine(ml, rl):
                correctLines += 1
                
    print(f"total : {len(modelLines)}, correct : {correctLines}")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modelOutFile")
    parser.add_argument("refFile")

    args = parser.parse_args()

    testFile(args.modelOutFile, args.refFile)