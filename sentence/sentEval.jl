using OhMyREPL
using PyCall

@pyimport importlib.machinery as machinery
loader = machinery.SourceFileLoader("SentEval","/home/PhD/github/SentEval/senteval/__init__.py")
sentEval = loader[:load_module]("SentEval")


PATH_TO_SENTEVAL = "/home/PhD/github/SentEval/"
PATH_TO_DATA = "../data"
PATH_TO_VEC = "/home/PhD/github/DATA/FT/sentEVAL/crawl-300d-2M.vec"

function create_dictionary(sentences; threshold=0)
    words = Dict{String, Int64}();
    for s in sentences
        for word in split(s)
            words[word] = get!(words, word, 0) + 1
        end
    end

    if threshold > 0
        newwords = Dict{String, Int64}();
        for word in words
            if words[word] >= threshold
                newwords[word] = words[word]
            end
        end
        words = newwords
    end
    words["<s>"] = 1e9 + 4
    words["</s>"] = 1e9 + 3
    words["<p>"] = 1e9 + 2

    sorted_words = sort(words, byvalue=true, rev=true)
    id2word = String[]
    word2id = Dict{String, Int64}()

    for (i, w) in enumerate(sorted_words)
        push!(id2word, w[1])
        push!(word2id, w[1] => i)
    end

    return id2word, word2id

end




# setting parameters for SentEval
params_senteval = Dict("task_path" => PATH_TO_DATA, "usepytorch" => true, "kfold" => 5)
params_senteval['classifier'] = Dict("nhid" => 0, "optim" => "rmsprop", "batch_size" => 128,
                                 "tenacity" => 3, "epoch_size" => 2)

se = sentEval.SE[:engine](params_senteval, batcher, prepare)
transfer_tasks = ["STS12", "STS13"]
results = se.eval(transfer_tasks)




py"""
from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging

# Set PATHs
PATH_TO_SENTEVAL = '/home/PhD/github/SentEval/'
PATH_TO_DATA = '/home/PhD/github/SentEval/data'
# PATH_TO_VEC = 'glove/glove.840B.300d.txt'
PATH_TO_VEC = '/home/PhD/github/DATA/FT/sentEVAL/crawl-300d-2M.vec'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


# SentEval prepare and batcher
def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    params.wvec_dim = 300
    return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
se = senteval.engine.SE(params_senteval, batcher, prepare)
"""
transfer_tasks = ["STS12", "STS13"]

py"se.eval"(transfer_tasks)
