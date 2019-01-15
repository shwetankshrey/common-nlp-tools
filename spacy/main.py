import spacy

nlp = spacy.load('en')

print("Enter text : ", end = "")
inp = input()

print("Enter task to perform : ", end = "")
chc = input()

# LEMMATIZATION, POS TAGGING

if chc == "1":
    doc = nlp(inp)
    print("TEXT\t\tLEMMA\t\tPOS")
    print()
    for token in doc:
        print(token.text + "\t\t" + token.lemma_ + "\t\t" + token.tag_)
    print()

# NAMED ENTITY RECOGNITION

if chc == "2":
    doc = nlp(inp)
    print("ENTITY\t\tLABEL")
    print()
    for ent in doc.ents:
        print(ent.text + "\t\t" + ent.label_)
    print()

# SIMILARITY

if chc == "3":
    word1, word2 = inp.split()
    d1 = nlp(word1)
    d2 = nlp(word2)
    print("Similarity : " + str(d1[0].similarity(d2[0])))


