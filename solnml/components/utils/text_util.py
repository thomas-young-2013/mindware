import numpy as np


def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower:
        text = text.lower()

    translate_dict = dict((c, split) for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]


def build_embeddings_index(glove_path='./glove_data/glove.6B.50d.txt'):
    embeddings_index = dict()
    f = open(glove_path, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def load_embedding_matrix(word_index, embedding_index, if_normalize=True):
    all_embs = np.stack(embedding_index.values())
    if if_normalize:
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
    else:
        emb_mean, emb_std = 0, 1
    embed_size = all_embs.shape[1]
    nb_words = len(word_index)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i - 1] = embedding_vector
    return embedding_matrix


def load_text_embeddings(texts, embedding_index, method='average', alpha=1e-3):
    from keras.preprocessing.text import Tokenizer
    tok = Tokenizer()
    tok.fit_on_texts(texts)
    word_index = tok.word_index
    embedding_matrix = load_embedding_matrix(word_index, embedding_index, if_normalize=False)
    index_sequences = tok.texts_to_sequences(texts)
    text_embeddings = []
    if method == 'average':
        for seq in index_sequences:
            embedding = []
            for i in seq:
                embedding.append(embedding_matrix[i - 1])
            text_embeddings.append(np.average(embedding, axis=0))
        text_embeddings = np.array(text_embeddings)
    elif method == 'weighted':
        from collections import Counter
        for seq in index_sequences:
            counter = Counter(seq)
            embedding = []
            for i in counter:
                embedding.append(embedding_matrix[i - 1] * alpha / (alpha + counter[i] / len(seq)))
            text_embeddings.append(np.average(embedding, axis=0))
        text_embeddings = np.array(text_embeddings)

        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
        svd.fit(text_embeddings)
        pc = svd.components_
        test_embeddings = text_embeddings - text_embeddings.dot(pc.transpose()) * pc
    return text_embeddings