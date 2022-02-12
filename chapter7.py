
# https://kakedashi-engineer.appspot.com/2020/05/09/nlp100-ch7/
import gensim



# #60. Loading word vectors
model = gensim.models.KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin',binary=True)

# print(model['United_States'])

# #61. word similarity
# print(model.similarity('United_States','U.S.'))

# #62. Top 10 most similar words
# print(model.most_similar('Unites_States',topn=10))

# # 63. Analogy based on additive composition  Spain-Madrid+Athens
# print(model.most_similar(positive=['Spain','Athens'], negative=['Madrid'],topn=10))

#64. analogy data experient
# vec(word in second column) - vec(word in first column) + vec(word in third column)
# file= open('questions-words.txt','r')
# save_F=open('similar-questions-words.txt','w')
# for line in file:
#     line = line.split()
#     cat=''
#     if line[0] == ':':
#         category = line[1]
#         print(category)
#     else:
#         # print(model.most_similar(positive=[line[1], line[2]], negative=[line[0]], topn=1))
#         word, cos = model.most_similar(positive=[line[1], line[2]], negative=[line[0]], topn=1)[0]
#         save_F.write(' '.join([category] + line + [word, str(cos) + '\n']))

#65. accuracy score on analogy task
sem_count,sem_ok,syn_count,syn_ok=0,0,0,0,
ok = 0
readF= open('similar-questions-words.txt','r')
for line in readF:
    line = line.split()
    if not line[0].startswith('gram'):
        sem_count+=1
        if line[4]==line[5]:
            sem_ok+=1
    else:
        syn_count+=1
        if line[4]==line[5]:
            syn_ok+=1
print('semantic accuracy',sem_ok/sem_count)
print('synatic accuracy',syn_ok/syn_count)


#66. evalution on word similarity- 353 spearman correlation
import pandas as pd
df=pd.read_csv('wordsim353/combined.csv')
# sim=[]
# for i in range(len(df)):
#     line = df.iloc[i]
#     sim.append(model.similarity(line['Word 1'],line['Word 2']))
# df['w2v'] = sim
# print(df[['Human (mean)', 'w2v']].corr(method='spearman'))


#67. k-means clustering
from sklearn.cluster import KMeans
import numpy as np

file= open('questions-words.txt','r')
countries= []
for line in file:
    line=line.split()
    print(line)
    if line[0] in ['capital-common-countries', 'capital-world']:
        countries.append(line[2])
    # elif line[0] in ['currency', 'gram6-nationality-adjective']:
    #     countries.add(line[1])


countries=list(set(countries))
print(countries)
countries_vector = [model[country] for country in countries]

# kmeans = KMeans(n_clusters=5, random_state=0)
# kmeans.fit(countries_vector)
# for i in range(5):
#     cluster = np.where(kmeans.labels_ == i)[0]
#     print('cluster', i)
#     print(', '.join([countries[k] for k in cluster]))
#
#
#
# # #68 Ward's method clustering
# import matplotlib.pyplot as plt
# from scipy.cluster.hierarchy import dendrogram, linkage
# plt.figure(figsize=(15,5))
# link = linkage(countries_vector, method='ward')
# dendrogram(link, labels=countries)
# # plt.show()
# plt.savefig('68.wardcustering.png')
#
# # #69. t-SNE visualization
# from sklearn.manifold import TSNE
# vec_embedded = TSNE(n_components=2).fit_transform(countries_vector)
# vec_embedded_t = list(zip(*vec_embedded))
# fig, ax = plt.subplots(figsize=(10, 10))
# plt.scatter(*vec_embedded_t)
# for i, c in enumerate(countries):
#     ax.annotate(c, (vec_embedded[i][0],vec_embedded[i][1]))
# plt.savefig('69.tsne-visualization.png')
