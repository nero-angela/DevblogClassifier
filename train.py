import os
from absl import app
from flags import create_flags, FLAGS
from word_embedding import WordEmbedding

def main(_):
    # create base directory
    for path in ['./data', './cf_model', './we_model']:
        if not os.path.isdir(path):
            os.makedirs(path)
    
    we = WordEmbedding()

    # word embedding model
    if FLAGS.we_model == 'devblog':
        we.loadDevblogModel(1, 2, 3, 4, 5)
        print('devblog') 

    elif FLAGS.we_model == 'wiki':
        print('wiki')

    # train
    if FLAGS.train:
        print('train')

    # predict
    if FLAGS.predict != None:
        print('predict')
    
#     doc = Document()
#     we = WordEmbedding()
#     cf = Classifier()
#     we_model = we.loadWikiModel()
#     model = cf.loadModel()
    
if __name__ == '__main__':
    create_flags()
    print('hello')
    app.run(main)
