# Devblog Classifier
[Awesome Devblog](https://github.com/sarojaba/awesome-devblog)에서 제공받은 데이터를 이용하여 제작한 개발 문서 분류기입니다.

---
## How To Run
- 개발환경
  - python 3.7
  - tensorflow 2.0.0
  - jupyter notebook

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
  - [labeled data](https://drive.google.com/drive/u/0/folders/1Npfrh6XmeABJ8JJ6ApS1T88vVoqyDH7M) : 23.5MB
  ~~~
  전체 데이터 수           : 34,620개
  라벨링 된 데이터 수       : 10,382개
  라벨링 안된 데이터 수      : 24,238개 (label -1)
  개발과 관련 있는 데이터 수  : 7,634개  (label  0)
  개발과 관련 없는 데이터 수  : 2,748개  (label  1)
  ~~~
  
  - preprocessing
  ~~~
  tags / 배열로 되어있으므로 띄어쓰기로 join
  title, description, tags / 영어, 한글, 공백만 남김
  html tag 삭제
  \n, \r 삭제
  2회 이상의 공백은 하나로 줄입
  영어 대문자 소문자로 변환
  앞뒤 공백 삭제
  블랙리스트 데이터 제외
  ~~~

  - 문서 대표 문장(text)
  ~~~
  text = tags + title + description
  ~~~

- word embedding
  - [wiki 한국어 데이터](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz) : 4.49GB 
  - wiki 한국어 데이터 기반 FastText model : 14.5GB
  - vocaburary size : 2,000,000
  - 단어 임베딩 모델 성능
  ~~~
  $ getSimilarWords('파이썬') # 유사한 단어 조회
  [('Python', 0.565061628818512),
  ('자이썬', 0.5624369382858276),
  ('레일스', 0.5598082542419434),
  ('파이썬을', 0.5595801472663879),
  ('언어용', 0.5288202166557312)]
  ~~~

- classifier
  - tensorflow model : 170KB
  - `Dense(100) -> Dense(80) -> Dense(2)`
  - performance
  ~~~
  loss : 0.278
  accuracy : 0.883
  f1_score : 0.847
  precision : 0.739
  recall : 1.0
  ~~~