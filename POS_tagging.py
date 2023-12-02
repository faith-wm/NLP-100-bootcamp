from collections import defaultdict
import re

txt_file='alice.txt'
conll_fil='alice.txt.conll'

text='hello.there'
print(re.search('[^A-Za-z0-9\s\w]',text))

nouns_l= ['NN', 'NNS', 'NNP', 'NNPS']
verbs_l= ['VB', 'VBD', 'VBG', 'VBN', 'VBP','VBZ']
def read_file():
    conll_fil_read=open(conll_fil,'r')
    result_dict=[]
    sentences=[]
    for line in conll_fil_read:
        # print(line)

        if line !='\n':
            attribute=line.split('\t')
            sent_dict={'text':attribute[1],'lemma':attribute[2],'pos':attribute[3]}
            result_dict.append(sent_dict)

        else:
            sentences.append(result_dict)
            result_dict=[]


    return sentences

sentences=read_file()
print(sentences[10])


print('\n -------------------getting verbs surface form-----------------')
verbs=[]
for sentence in sentences:
    for dict in sentence:
        if dict['pos']=='VB':
            verbs.append(dict['text'])
print(verbs)



print('\n -------------------getting verbs lemma-----------------')
lemmas=[]
for sentence in sentences:
    for dict in sentence:
        if dict['pos']=='VB':
            lemmas.append(dict['lemma'])
print(verbs)

print(lemmas)



print('\n ------------------noun phrases in form A of B-----------------')
A_ofB=[]
for sentence in sentences:
    for i in range(1,len(sentence)-1):
        if sentence[i-1]['pos']=='NN' and sentence[i]['text']=='of' and sentence[i+1]['pos']=='NN':
            A_ofB.append(sentence[i-1]['text'] +' '+ sentence[i]['text'] +' ' + sentence[i+1]['text'])

print(A_ofB)

print('\n ------------------AB longest noun phrase-----------------')
AB=[]
nouns=''
num=0
for sentence in sentences:
    for i in range(len(sentence)):
        if sentence[i]['pos']=='NN':
            nouns= nouns+ ' '+ sentence[i]['text']
            num+=1
        elif num>=2:
            AB.append(nouns)
            nouns=''
            num=0
        else:
            nouns=''
            num=0

print(AB)


print('\n ------------------getting word frequecy-----------------')
import string
punctuation=string.punctuation
words_count={}
for sentence in sentences:
    for i in range(len(sentence)):
        if sentence[i]['text'] not in punctuation:
            if sentence[i]['text'] not in words_count:
                words_count[sentence[i]['text']]=1
            else:
                words_count[sentence[i]['text']] += 1

words_count=sorted(words_count.items(), key=lambda x:x[1], reverse=True)
print(words_count)


print('\n ------------------top 10 frequenct words-----------------')
import matplotlib.pyplot as plt
most_frequent= [i[0] for i in words_count[:10]]
values=[i[1] for i in words_count[:10]]

print(list(zip(most_frequent,values)))

plt.bar(most_frequent,values)
plt.show()


print('\n ------------------top 10 co-occuring with Alice-----------------')
co_occuring={}
for sentence in sentences:
    if 'Alice' in [dict['text'] for dict in sentence]:
        for i in range(len(sentence)):
            if sentence[i]['text'] not in punctuation:
                if sentence[i]['text'] not in co_occuring:
                    co_occuring[sentence[i]['text']] =1
                else:
                    co_occuring[sentence[i]['text']] += 1
del co_occuring['Alice']


co_occuring=sorted(co_occuring.items(), key=lambda x:x[1], reverse=True)
print(co_occuring)

x= [i[0] for i in co_occuring[:10]]
y=[i[1] for i in co_occuring[:10]]

plt.bar(x,y)
plt.show()


print('\n ------------------word frequency histogram-----------------')
punctuation=string.punctuation
words_count={}
for sentence in sentences:
    for i in range(len(sentence)):
        if sentence[i]['text'] not in punctuation:
            if sentence[i]['text'] not in words_count:
                words_count[sentence[i]['text']]=1
            else:
                words_count[sentence[i]['text']] += 1
x=words_count.values()
plt.hist(x,bins=100)
plt.show()



print('\n ------------------zipfs law-----------------')
import math
words_count=sorted(words_count.items(), key=lambda x:x[1], reverse=True)

x= [math.log(r + 1) for r in range(len(words_count))]
y = [math.log(a[1]) for a in words_count]

plt.figure(figsize=(8, 4))
plt.scatter(x, y)
plt.show()
