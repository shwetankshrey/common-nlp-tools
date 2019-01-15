import gensim
import os
import scipy

model = gensim.models.Doc2Vec.load("model.doc2vec")

rootdir = "20_newsgroups"
maindir = "comp.graphics"
otherdirs = ["alt.atheism","comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med", "sci.space", "soc.religion.christian", "talk.politics.guns", "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc"]

# DIFFERENT CLASSES

mean_diff = 0
file = os.listdir(rootdir + "/" + maindir)[0]
fl = open(rootdir+"/"+maindir+"/"+file, encoding="iso-8859-1")
txt = fl.read()
fl.close()
v1 = model.infer_vector(gensim.utils.simple_preprocess(txt))
for dir in otherdirs:
    file = os.listdir(rootdir + "/" + dir)[0]
    fl = open(rootdir+"/"+dir+"/"+file, encoding="iso-8859-1")
    txt = fl.read()
    fl.close()
    v2 = model.infer_vector(gensim.utils.simple_preprocess(txt))
    similarity = 1 - scipy.spatial.distance.cosine(v1, v2)
    mean_diff += similarity
mean_diff /= 19

# SAME CLASS

mean_same = 0
file = os.listdir(rootdir + "/" + maindir)[0]
fl = open(rootdir+"/"+maindir+"/"+file, encoding="iso-8859-1")
txt = fl.read()
fl.close()
v1 = model.infer_vector(gensim.utils.simple_preprocess(txt))
for i in range(1, 20):
    file = os.listdir(rootdir + "/" + maindir)[i]
    fl = open(rootdir+"/"+maindir+"/"+file, encoding="iso-8859-1")
    txt = fl.read()
    fl.close()
    v2 = model.infer_vector(gensim.utils.simple_preprocess(txt))
    similarity = 1 - scipy.spatial.distance.cosine(v1, v2)
    mean_same += similarity
mean_same /= 19

print("Normalized similarity with different classes    :    " + str(mean_diff))
print("Normalized similarity when same class           :    " + str(mean_same))
