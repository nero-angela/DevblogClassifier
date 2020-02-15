from util import downloadByURL
from gensim.models import FastText, fasttext

"""
FastText base word embedding
"""
class WordEmbedding():
    
    def __init__(self):
        # corpus
        self.WIKI_KO_DATA = './data/cc.ko.300.bin.gz'
        self.WIKI_KO_DATA_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz'

        # pretrained model
        self.WIKI_KO_MODEL_PATH = f'./wv_model/ko.wiki'

    def getCustomModel(self, text, size=4, window=3, min_count=1, epochs=10):
        """
        주어진 text 기반 word embedding 생성
        """
        model = FastText(size=size, window=window, min_count=min_count)
        model.build_vocab(sentences=text)
        model.train(sentences=text, total_examples=len(text), epochs=epochs)
        return model
    
    def getWikiModel(self):
        """
        Load wiki corpus base pretrained fasttext model
        """
        model = None
        if not os.path.isfile(self.WIKI_KO_MODEL_PATH):
            print('학습된 ko_wiki 모델이 없습니다.')
            
            if not os.path.isfile(self.WIKI_KO_DATA):
                print('학습에 필요한 WIKI_KO_DATA가 없으므로 다운로드를 시작합니다.')
                downloadByURL(self.WIKI_KO_DATA_URL, self.WIKI_KO_DATA)
            
            print('WIKI_KO_DATA 기반으로 모델을 생성합니다.')
            model = fasttext.load_facebook_model(self.WIKI_KO_DATA)
            model.save(self.WIKI_KO_MODEL_PATH)
            
        else:
            model = FastText.load(self.WIKI_KO_MODEL_PATH)
        
        print(f'vocab size : {len(model.wv.vocab)}')
        return model
    
    def getSimilarWords(self, model, word, topn=5):
        return model.wv.similar_by_word(word, topn)
