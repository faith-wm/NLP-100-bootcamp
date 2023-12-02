#https://qiita.com/yamaru/items/255d0c5dcb2d1d4ccc14
import json
import re

file='enwiki-country.json'

def readDoc(file):
    doc = [json.loads(line) for line in open(file, 'r')]
    return doc



def getUK(doc):
    for article in doc:
        key = article.keys()  # dict_keys(['title', 'text'])
        val = article.values()
        if article['title'] == 'United Kingdom':
            text = article['text']
            break
    return text


doc=readDoc(file)
UK_article=getUK(doc)
print(UK_article)


print('======================================\n')
def getCategoryLines(article):
    pattern='^(.*\[\[Category:.*\]\].*)$'
    category_names=re.findall(pattern,article, re.MULTILINE)
    return '\n'.join(category_names)

print(getCategoryLines(UK_article))


print('======================================\n')
def getCategoryNames(article):
    pattern = '^.*\[\[Category:(.*?)\]\].*$'
    category_names = re.findall(pattern, article, re.MULTILINE)

    # pattern = '^.*\[\[Category:(.*?)(?:\|.*)?\]\].*$'
    # category_names = re.findall(pattern, article, re.MULTILINE)

    return '\n'.join(category_names)

print(getCategoryNames(UK_article))



print('======================================\n')
def getSections(article):
    pattern='^\={2,}(.*?)\={2,}.*$'
    sec_names = re.findall(pattern, article, re.MULTILINE)
    return '\n'.join(sec_names)

print(getSections(UK_article))



print('======================================\n')
def getMediaRef(article):
    pattern = '\[\[File:(.*?)\|'
    mediaRef_names = re.findall(pattern, article, re.MULTILINE)

    return '\n'.join(mediaRef_names)

print(getMediaRef(UK_article))



print('======================================\n')
def getInfobox(article):
    pattern = '^\{\{Infobox.*?$(.*?)^\}\}'
    infobox = re.findall(pattern, article, re.MULTILINE+re.DOTALL)
    # for i in infobox:
    #     print(i)

    # print('======================================\n')
    pattern = r'^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))'
    result = dict(re.findall(pattern, infobox[0], re.MULTILINE + re.DOTALL))  # DOTALL flag has been specified, this matches any character including a newline.

    return result

info_box=getInfobox(UK_article)
for k, v in info_box.items():
    print(k + ': ' + v)



print('======================================\n')
def removeEmphasisMarkups(text):
    pattern='\'{2,5}'
    return re.sub(pattern,'',text)

remove_emphasis={k: removeEmphasisMarkups(v) for k,v in info_box.items()}
for k, v in remove_emphasis.items():
    print(k + ': ' + v)


print('======================================\n')
def removeLinks(text):
    #remove emphsis markups
    pattern = '\'{2,5}'
    text=re.sub(pattern, '', text)

    #remove links
    pattern = r'\[\[(?:[^|]*?\|)??([^|]*?)\]\]'
    text = re.sub(pattern, r'\1', text)
    return text

remove_links={k: removeLinks(v) for k,v in info_box.items()}
for k, v in remove_links.items():
    print(k + ': ' + v)


print('======================================\n')
def removeMediaMarkup(text):
    #remove marups
    pattern = r'\'{2,5}'
    text = re.sub(pattern, '', text)

    # remove link
    pattern = r'\[\[(?:[^|]*?\|)??([^|]*?)\]\]'
    text = re.sub(pattern, r'\1', text)

    # remove markup
    pattern = r'https?://[\w!?/\+\-_~=;\.,*&@#$%\(\)\'\[\]]+'
    text = re.sub(pattern, '', text)

    # remove htnl tag
    pattern = r'<.+?>'
    text = re.sub(pattern, '', text)

    # # remove template
    # pattern = r'\{\{(?:lang|仮リンク)(?:[^|]*?\|)*?([^|]*?)\}\}'
    # text = re.sub(pattern, r'\1', text)
    return text

remove_media_markup={k: removeMediaMarkup(v) for k,v in info_box.items()}
for k, v in remove_media_markup.items():
    print(k + ': ' + v)



print('======================================\n')
def getFlag_url(doc):
    # url_file = text['国旗画像'].replace(' ', '_')
    # url = 'https://commons.wikimedia.org/w/api.php?action=query&titles=File:' + url_file + '&prop=imageinfo&iiprop=url&format=json'
    # data = requests.get(url)
    # return re.search(r'"url":"(.+?)"', data.text).group(1)
    return 0
