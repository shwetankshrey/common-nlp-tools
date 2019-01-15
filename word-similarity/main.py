import gensim

model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

# If Delhi is the capital of India then what is the capital of China?

result = model.most_similar(positive=['Delhi', 'China'], negative=['India'])
print("Delhi - India + China : ")
print(result)
print()
print()

# If ISRO is related to India then what is related to USA?

result = model.most_similar(positive=['ISRO', 'USA'], negative=['India'])
print("ISRO - India + USA : ")
print(result)
print()
print()
