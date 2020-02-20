import os, re
from absl import flags

def create_constants():
    class CONST(object):
        pass
    CONST = CONST()

    # awesome devblog
    CONST.origin_data_url = 'https://awesome-devblog.now.sh/api/korean/people/feeds'
    CONST.origin_max_req_size = 5000

    # devblog
    CONST.devblog_data_url = 'https://drive.google.com/uc?id=1K5Isidyb1O7OXQ47Yk2fMVYBvEoL6W4-&export=download'
    CONST.devblog_data_path = './data/documents.csv'
    CONST.devblog_model_path = './we_model/devbog'
    
    # wiki
    CONST.wiki_data_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz'
    CONST.wiki_data_path = './data/cc.ko.300.bin.gz'
    CONST.wiki_model_path = './we_model/wiki'

    # classifier
    CONST.classifier_model_path = './cf_model/classifier'
    return CONST

def common_flags(f):
    # word embedding
    f.DEFINE_integer('we_dim', 300, 'word embedding dimension')
    f.DEFINE_integer('we_epoch', 3, 'word embedding epoch')
    f.DEFINE_integer('we_window', 3, 'word embedding window size')
    f.DEFINE_integer('we_min_count', 3, 'word embedding min count')

    # classifier
    f.DEFINE_string('cf_checkpoint', './checkpoint', 'classifier model checkpoint path')

def create_predict_flags():
    f = flags
    common_flags(f)

    # predict
    f.DEFINE_string('predict', None, "sentence you want to predict")

    # word embedding
    f.DEFINE_enum('we_model', 'devblog', ['wiki', 'devblog'], 'word embedding model you want use')
    
    # validation
    f.register_validator('predict',
                         lambda x: x != None,
                         message="write the sentence you want to predict. ex) 'how to learn python'")

def create_train_flags():
    f = flags
    common_flags(f)

FLAGS = flags.FLAGS
CONST = create_constants()