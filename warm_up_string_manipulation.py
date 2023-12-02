def reverseString(text):
    return text[::-1]

print(reverseString('stressed'))

def schooled(text):
    return text[1]+text[3]+text[5]+text[7]

print(schooled('schooled'))

def stringConcatenate():
    s1='shoe'
    s2='cold'
    output=''
    for i in range(len(s1)):
        output+=s1[i]
        output+=s2[i]
    return output
print(stringConcatenate())

def pi():
    sentence='Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics'
    sentence=sentence.replace(',','')
    words=sentence.split(' ')
    output=[len(w) for w in words]
    return output
print(pi())

def atomicSymbols():
    sentence='Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can'
    sentence = sentence.replace(',', '')
    sentence = sentence.replace('.', '')
    words = sentence.split(' ')

    oneword=[1,5,6,7,8,9,15,16,19]
    oneword=[i-1 for i in oneword]

    dict_output={}
    for i in range(len(words)):
        j=i+1
        if i in oneword:
            dict_output[j]=words[i][0]
        else:
            dict_output[j]=words[i][:2]
    return dict_output
print(atomicSymbols())


def char_ngram(text,n):
    # text=text.replace(' ','')
    grams=[text[i:i+n] for i in range(len(text)-n+1)]
    return grams
print(char_ngram('I am an NLPer',2))

def word_ngram(text,n):
    text=text.split(' ')
    grams=[text[i:i+n] for i in range(len(text)-n+1)]
    return grams

print(word_ngram('I am an NLPer',2))


def bigram_set():   #set {}, store multiple items, it is unordered, unindexed, no duplicates, unchangeable
    x=set(char_ngram('paraparaparadise',2))
    y=set(char_ngram('paragraph', 2))

    union=x.union(y)  #union- all items both
    intersection=x.intersection(y) #shared
    difference=union-intersection

    diff2=[b for b in x if b not in y]+[b for b in y if b not in x]

    print('union, intersection, difference')
    print(x),print(y)
    print(union), print(intersection), print(difference), print(diff2)
    return 0

print(bigram_set())
# from nltk.util import ngrams
# print(list(ngrams('paragraph',2)))

def template_sente_gen(x,y,z):
    return f'{y} is {z} at {x}'#'{} is {} at {}'.format(y,z,x)
print(template_sente_gen(x=12,y='temperature',z=22.4))


def cipherText(text):
    # to get ascii code use ord() function, chr returns string for unicode code
    cipheredText=''
    for c in text:
        if c.islower():
            cipheredText+=chr(219-ord(c))
        else:
            cipheredText+=c
    return cipheredText
print('cipher: ', cipherText('hello'), cipherText(cipherText('hello')))

def typoglycemia(text):
    words=text.split(' ')
    output=[]
    for w in words:
        if len(w)<=4:
            output.append(w)
        else:
            new_w=w[0]+w[len(w)-1:0:-1]+w[-1]
            output.append(new_w)
    return ' '.join(output)
print(typoglycemia('I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind'))

print('\n\n')
print(ord('a'))
print(chr(ord('a')))
