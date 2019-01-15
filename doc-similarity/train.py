import os
import gensim

rootdir = "20_newsgroups"
maindir = "comp.graphics"
otherdirs = ["alt.atheism","comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med", "sci.space", "soc.religion.christian", "talk.politics.guns", "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc"]

training_corpus = []

count = 0
for dir in otherdirs + [maindir]:
    files = os.listdir(rootdir + "/" + dir)
    for file in files:
        fl = open(rootdir+"/"+dir+"/"+file, encoding="iso-8859-1")
        txt = fl.read()
        fl.close()
        training_corpus.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(txt), [count]))
        count += 1

print(training_corpus[:3])
print(len(training_corpus))

model = gensim.models.doc2vec.Doc2Vec(vector_size=108, min_count=2, epochs=40)
model.build_vocab(training_corpus)
model.train(training_corpus, total_examples=model.corpus_count, epochs=model.epochs)

model.save("model.doc2vec")

