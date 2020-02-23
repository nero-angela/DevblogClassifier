import os
import pandas as pd
import numpy as np
from absl import app
from util import han2Jamo
from flags import create_flags, FLAGS, CONST
from word_embedding import WordEmbedding
from classifier import Classifier
from document import Document

def train(_):

    # init
    we = WordEmbedding()
    dc = Document()
    cf = Classifier()

    # load data
    docs = dc.getDocs(labeled_only=True)

    # load word embedding model
    if FLAGS.we_model == 'devblog':
        we_model = we.loadDevblogModel(embedding_dim = FLAGS.we_dim,
                                       epochs        = FLAGS.we_epoch,
                                       window        = FLAGS.we_window,
                                       min_count     = FLAGS.we_min_count)
        # han2jamo
        docs.text = docs.text.apply(han2Jamo)
    elif FLAGS.we_model == 'wiki':
        we_model = we.loadWikiModel()

    # word embedding
    docs.vector = docs.text.apply(lambda x: we.embedding(we_model, x))

    # training
    cf_model = cf.train(docs, './checkpoint')
    cf.saveModel(cf_model, FLAGS.cf_model)
    
if __name__ == '__main__':
    create_flags()
    app.run(train)
