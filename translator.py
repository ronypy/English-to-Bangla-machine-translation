import json
import os

def tokenize(sentence):
    return sentence.split()


def translate(tokens, model):
    return [model[word] if word in model else word for word in tokens]


def main():
    workingDir = os.getcwd()
    sentenceFile = workingDir + '/data/Testing_model.txt'
    model_file = workingDir + '/data/Translation_model.txt'
    output_file = workingDir + '/data/Output_result.txt'
    out_write = open(output_file,'w')
    with open(model_file, 'r') as f:
        model = json.load(f)
    sf = open(sentenceFile,'r')
   
    for sentence in sf.readlines():
        outStr = ' '
        tokens = tokenize(sentence)
        translated_tokens = translate(tokens, model)
        outStr = outStr.join(translated_tokens)
        out_write.write('English Sentence: ' + sentence + 'Bangla Translation: ' + outStr.encode('utf-8') + '\n\n' )
        

if __name__ == '__main__':
    main()
