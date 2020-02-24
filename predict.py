import os
import pandas as pd
import numpy as np
from absl import app
from util import han2Jamo
from flags import create_flags, FLAGS, CONST
from word_embedding import WordEmbedding
from classifier import Classifier
from document import Document

def main(_):
    # init
    we = WordEmbedding()
    dc = Document()
    cf = Classifier()

    # load word embedding model
    if FLAGS.we_model == 'devblog':
        we_model = we.loadDevblogModel(embedding_dim = FLAGS.we_dim,
                                       epochs        = FLAGS.we_epoch,
                                       window        = FLAGS.we_window,
                                       min_count     = FLAGS.we_min_count)
    elif FLAGS.we_model == 'wiki':
        we_model = we.loadWikiModel()

    # load classifier model
    cf_model = cf.loadModel(FLAGS.cf_model)

    results = [{'text': r} for r in FLAGS.predict]
    is_devblog = FLAGS.we_model == 'devblog'
    for i, r in enumerate(FLAGS.predict):
        # preprocessing
        text = han2Jamo(r) if is_devblog else r

        # word embedding
        df = dc.preprocessing(text, devblog=is_devblog)
        vector = df.text.apply(lambda x: we.embedding(we_model, x, FLAGS.we_dim)).tolist()
        if len(vector) == 0:
            print('🐈 text is not valid')
            return
        else:
            # predict
            results[i]['predict'] = cf.predict(cf_model, np.array(vector))
    
    return results
    
if __name__ == '__main__':
    create_flags(True)
    app.run(main)
