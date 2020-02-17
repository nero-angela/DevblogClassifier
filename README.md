# Devblog Classifier
- [Daily Devblog](http://daily-devblog.com/) 서비스에 개발 이외의 글을 필터링하기 위한 문서 분류기입니다.
- [Awesome Devblog](https://github.com/sarojaba/awesome-devblog)에서 제공받은 데이터를 이용하였습니다.

---
## How To Run
- 개발환경
  - Python 3.7
  - Tensorflow 2.0.1
  - Gensim FastText

- 라이브러리 설치
  ~~~
  $ pip install -r requirements.txt
  ~~~

- README.ipynb 참고
  ![how to run](https://user-images.githubusercontent.com/26322627/74600924-5e65a100-50db-11ea-8dad-31d18d909053.png)

---
## How To Training
- train
  ~~~
  from main import *
  train()
  ~~~

- 학습에 필요 용량 : 19GB
  - [labeled data](https://drive.google.com/drive/u/0/folders/1Npfrh6XmeABJ8JJ6ApS1T88vVoqyDH7M) : 23.5MB
  - [wiki 한국어 데이터](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz) : 4.49GB 
  - wiki 한국어 데이터 기반 FastText model : 14.5GB
  - tensorflow model : 170KB

- data
  - `tags / title / description` 컬럼 사용
  - analysis
  ~~~
  - 데이터 수량 조사
  Total data : 34,620개
  Total labeled data : 10,382개
  [label -1] 라벨링 안된 데이터 수 : 24,238개
  [label  0] 개발과 관련 있는 데이터 수 : 7,634개
  [label  1] 개발과 관련 없는 데이터 수 : 2,748개

  - 문장 길이 조사
  문장 길이 최대 값 : 358
  문장 길이 최소 값 : 3
  문장 길이 평균 값 : 145.02
  문장 길이 표준편차 : 60.59
  문장 길이 중간 값 : 132.0
  문장 길이 제 1 사분위 : 108.0
  문장 길이 제 3 사분위 : 203.0
  ~~~
  ![text length](https://user-images.githubusercontent.com/26322627/74600892-e4351c80-50da-11ea-9454-5397bf134ace.png)

  - text word cloud
  ![word cloud](https://user-images.githubusercontent.com/26322627/74600889-dc757800-50da-11ea-9e55-97010103b606.png)
  
  - preprocessing
  ~~~
  - tags
  배열로 되어있으므로 띄어쓰기로 join
  
  - title, description, tags
  영어, 한글, 공백만 남김

  - _id, title, description, tag, link
  html tag 삭제
  \n, \r 삭제
  2회 이상의 공백은 하나로 줄입
  영어 대문자 소문자로 변환
  앞뒤 공백 삭제
  블랙리스트 데이터 제외

  - text 추가 (대표 문장)
  text = tags + title + description
  ~~~

  - 전처리 완료된 데이터 예시
  ~~~
  label : -1
  _id : 5e0415143e8fe000041459b2
  title : 한국의 파이썬 소식년 월 넷째 주
  description : 한국에서 일어나는 파이썬 관련 소식을 전합니다 알고리즘 시각화용 프로젝트 ipytracer 공개 미세먼지 대기정보 알림 봇 제작기 파이콘 년 월 세미나
  tags : algorithm python
  link : http://raccoonyy.github.io/python-news-for-korean-2017-4th-week-mar/
  text : algorithm python 한국의 파이썬 소식년 월 넷째 주 한국에서 일어나는 파이썬 관련 소식을 전합니다 알고리즘 시각화용 프로젝트 ipytracer 공개 미세먼지 대기정보 알림 봇 제작기 파이콘 년 월 세미나
  ~~~

- word embedding
  - wiki 한국어 데이터 기반 FastText model
  - vocaburary size : 2,000,000
  - embedding dimension : 300
  - wv_model 유사단어 조회
  ~~~
  $ getSimilarWords('파이썬') # 유사한 단어 조회
  [('Python', 0.565061628818512),
  ('자이썬', 0.5624369382858276),
  ('레일스', 0.5598082542419434),
  ('파이썬을', 0.5595801472663879),
  ('언어용', 0.5288202166557312)]
  ~~~

- classifier
  - tensorflow 2.0.1
  - layer
  ~~~
  model.add(Dense(100, activation='relu', kernel_initializer='he_normal')
  model.add(Dense(80, activation='relu', kernel_initializer='he_normal'))
  model.add(Dense(2, activation='softmax'))
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
  ~~~

  - performance
  ~~~
  loss : 0.278
  accuracy : 0.883
  f1_score : 0.847
  precision : 0.739
  recall : 1.0
  ~~~

  - train history
  ![train history](https://user-images.githubusercontent.com/26322627/74600880-d089b600-50da-11ea-95d4-ee22a7611dd6.png)
