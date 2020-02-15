import os
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

class Analysis():
     
    def __init__(self, data):
        """
        전체 분석
        """
        self.countAnalysis(data)
        print()
        self.textAnalysis(data)
        print()
        self.showWordCloud(data.text)
        
    def countAnalysis(self, data):
        """
        데이터 수량 조사
        """
        
        labeled_data = data.loc[data.label != -1]
        total_count = len(data) # 전체 데이터 수
        labeled_count = len(labeled_data) # 라벨링 된 데이터 수

        print('> 데이터 수량 조사')
        print(f'전체 데이터 수: {total_count}개')
        print(f'라벨링된 데이터 수: {labeled_count}개')
        for label, count in data.label.value_counts().iteritems():
            print(f'class {label} : {count}개')
    
    def textAnalysis(self, data):
        """
        text 길이 분석
        """
        text_len = data.text.apply(len)
        plt.figure(figsize=(12, 5))
        plt.hist(text_len, bins=200, alpha=0.5, color= 'r', label='length of text')
        plt.legend(fontsize='x-large')
        plt.yscale('log', nonposy='clip')
        plt.title('Log-Histogram of length of text')
        plt.xlabel('Length of text')
        plt.ylabel('Number of text')

        print('> 문장 길이 분석')
        print('문장 길이 최대 값: {}'.format(np.max(text_len)))
        print('문장 길이 최소 값: {}'.format(np.min(text_len)))
        print('문장 길이 평균 값: {:.2f}'.format(np.mean(text_len)))
        print('문장 길이 표준편차: {:.2f}'.format(np.std(text_len)))
        print('문장 길이 중간 값: {}'.format(np.median(text_len)))

        # 사분위의 대한 경우는 0~100 스케일로 되어있음
        print('문장 길이 제 1 사분위: {}'.format(np.percentile(text_len, 25)))
        print('문장 길이 제 3 사분위: {}'.format(np.percentile(text_len, 75)))
            
    def showWordCloud(self, text):
        """
        WordCloud
        """
        # 한글 폰트 깨짐방지
        for font in ["/Library/Fonts/NanumGothic.ttf", "/Library/Fonts/NotoSansCJKkr-Light.otf"]:
            if os.path.isfile(font):
                FONT_PATH = font
                break
        cloud = WordCloud(font_path=FONT_PATH).generate(" ".join(text))
        plt.figure(figsize=(20, 15))
        plt.imshow(cloud)
        plt.axis('off')
