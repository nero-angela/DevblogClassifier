class Classifier():
    
    def __init__(self):
        return
        
    def getWordEmbedding(self, text, size=4, window=3, min_count=1, epochs=10):
        """
        주어진 text 기반 word embedding 학습
        """
        model = FastText(size=size, window=window, min_count=min_count)
        model.build_vocab(sentences=text)
        model.train(sentences=text, total_examples=len(text), epochs=epochs)
        return model
    
    def getPretrainedWordEmbedding(self)
        
        
    def train(self):
        return
    
    def predict(self):
        return
        
dc = Classifier()
