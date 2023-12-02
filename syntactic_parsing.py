import stanza

json_file='ai.en.txt.json'
text_file='ai.en.txt'
my_parsed_file='parsed_file.txt'

print('\n ------------------parse result (words)-----------------')

# stanza.download('en',processors='tokenize,lemma,pos,depparse,sentiment,ner')
nlp=stanza.Pipeline(processors='tokenize,lemma,pos,depparse,sentiment,ner')
#
doc=nlp("Alice in wonderland.")
print(doc.sentences[0])
for sent in doc.sentences:
    print('--------------\n')
    print(sent.ents)
    print(sent.dependencies)
    print('--------------\n')
    for word in sent.words:
        # print(word)
        print(word.text,word.lemma,word.pos,word.deprel)


def parse_file(txt_file, output_file_name):
    read_txtfile=open(txt_file,'r')
    output_file=open(output_file_name,'w')
    for line in read_txtfile:
        doc=nlp(line)
        for sentence in doc.sentences:
            for word in sentence.words:
                output_file.write('{}\t{}\t{}\n'.format(word.text,word.lemma,word.pos))
            output_file.write('\n')


# parse_file(text_file,my_parsed_file)

class Word:
    def __init__(self,surface, lemma, pos):
        self.surface=surface
        self.lemma=lemma
        self.pos=pos
    def __str__(self):
        return 'surface: {}, lemma: {}, pos: {}'.format(self.surface,self.lemma,self.pos)

def sentence_list(parsed_file):
    sentences=[]
    sentence=[]
    read_file=open(parsed_file,'r')
    for line in read_file:
        if line != '\n':
            attribute = line.split('\t')
            sent_list =Word(surface=attribute[0], lemma=attribute[1], pos=attribute[2].replace('\n',''))
            sentence.append(str(sent_list))
        else:
            sentences.append(sentence)
            sentence=[]

    return sentences

sentences_array=sentence_list(my_parsed_file)
print(sentences_array[0])

print('\n ------------------parse result (dependency)-----------------')
