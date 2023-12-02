import stanza
import spacy
from spacy import displacy



json_file='ai.en.txt.json'
text_file='ai.en.txt'
my_parsed_file='spacy_parsed_file.txt'


print('\n ------------------parse result (words and dependency)-----------------')

nlp = spacy.load("en_core_web_sm")
doc=nlp('Frank Rosenblatt invented the perceptron')
for sent in doc.sents:
    print('-------------------------------\n')
    print(sent.text)
    for token in sent:
        print('surface: {}\n lemma: {}\n pos: {} \n head: {} \n dep: {}\n'.format(token.text, token.lemma_,
                                                    token.pos_, token.head.text, token.dep_))


def parse_file(txt_file, output_file_name):
    read_txtfile=open(txt_file,'r')
    output_file=open(output_file_name,'w')
    for line in read_txtfile:
        doc=nlp(line)
        for index,sentence in enumerate(doc.sents):
            for word in sentence:
                output_file.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(index,word.text,word.lemma_,word.pos_,
                                                               word.head.text, word.dep_))
            output_file.write('\n')


# parse_file(text_file,my_parsed_file)



class Word:
    def __init__(self,surface, lemma, pos,head, dep):
        self.surface=surface
        self.lemma=lemma
        self.pos=pos
        self.head= head
        self.dep = dep
    def __str__(self):
        return 'surface: {}, lemma: {}, pos: {}, head: {}, dep: {}'.format(self.surface,self.lemma,self.pos,
                                                                           self.head,self.dep.replace('\n',''))

def sentence_list(parsed_file):
    sentences=[]
    sentence=[]
    read_file=open(parsed_file,'r')
    for line in read_file:
        try:
            if line != '\n':
                attribute = line.split('\t')
                sent_list =Word(surface=attribute[1], lemma=attribute[2], pos=attribute[3], head=attribute[4],
                                dep=attribute[5])
                sentence.append(str(sent_list))
            else:
                sentences.append(sentence)
                sentence=[]
        except:
            continue

    return sentences

sentences_array=sentence_list(my_parsed_file)
print(sentences_array[0])

print('\n ------------------42: find root words-----------------')
read_txtfile=open(text_file,'r')
for line in read_txtfile:
    doc=nlp(line)
    count=0
    for sentence in doc.sents:
        print(count,'------------------------\n')
        count= count+1
        # print('\n',sentence)
        print('ROOT: ',sentence.root)
        print('CHILDREN: ',list(sentence.root.children))
    if count==1:
        break

print('\n ------------------43: governors and noun dependents-----------------')


print('\n ------------------44: visualizing -----------------')
# read_txtfile=open(text_file,'r')
# for line in read_txtfile:
#     doc=nlp(line)
#     count=0
#     for sentence in doc.sents:
#         print(sentence)
#         displacy.serve(sentence,style='dep')

print('\n ------------------45: triple -----------------')
read_txtfile=open(text_file,'r')
for line in read_txtfile:
    doc=nlp(line)
    count=0
    for sentence in doc.sents:
        print('\n')
        count= count+1
        tuple=[]
        for word in sentence:
            if word.dep_=='nsubj':
                print(word)
            if word.dep_=='ROOT':
                print(word)
            if word.dep_=='dobj':
                print(word)

    if count==1:
        break
