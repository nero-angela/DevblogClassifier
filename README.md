# Devblog Classifier
- 개발 관련 글인지 분류하는 어플리케이션입니다.
- [Awesome Devblog](https://github.com/sarojaba/awesome-devblog)에서 제공받은 데이터를 이용하였습니다.
- 개발 후기는 [딥러닝으로 문서 분류기 만들기](https://blog.devstory.co.kr/post/devblog-classifier/) 포스팅을 참고해 주세요.

## Demo
### 비개발 문서로 판단
```
(False, 0.191) 영어패턴#37
(False, 0.013) 당신은 정말 의지가 있는가?
(False, 0.041) 손흥민 시즌 13호골! epl 25라운드 토트넘 vs 맨체스터시티
(False, 0.051) 필리핀 세부 시티의 맛집! 하우스 오브 레촌 cebu city House of Lechon
(False, 0.191) 영어패턴#38
(False, 0.053) 언덕 너머
(False, 0.185) Ansible 파일 마지막 변경 일자 확인하기 😓
```

### 개발 문서로 판단
```
(True, 0.983) 맥북 초보자들이 꼭 알아야 할 단축키 top 5! mac keyboard shortcut 5. feat.맥북을 산  이유
(True, 1.000) [Spring Boot] 내장 웹 서버 - 2 (스프링부트 HTTPS / HTTP2)
(True, 0.971) 네이버클라우드플랫폼 Certificate Manager 에 LetsEncrypt 인증서 등록
(True, 0.791) Go로 블록체인 만들기 #1
(True, 1.000) 엔티티 매핑
(True, 0.973) [11775]Compactness criteria for clustering
(True, 0.989) 쿠버네티스 CI/DI 를 위한 오픈소스 프로젝트 알아보기
(True, 0.992) [Windows] USB 윈도우 10 설치 /  다운로드 방법
(True, 0.981) [Spring Boot] JAR파일(독립적으로 실행가능)
(True, 1.000) 케라스(Keras)의 get_file 함수
(True, 0.987) 웹 서비스 Maintenance Mode (점검 모드) 지원기
(True, 0.996) Nodejs AES 128 CCM 암호화(복호화) 예제 - crypto
(True, 0.706) [운영체제(OS)] 7. 쓰레드(Thread)
(True, 1.000) 비동기 처리와 콜백함수 그리고 Promise
(True, 0.994) docker기반 데이터 시각화툴 Superset 설치하기 (리눅스)
(True, 0.980) Wayland과 Weston
(True, 0.937) 테스트 주도 개발(Test-Driven Development:By Example) - 1부 : 화폐 예제 (9 ~ 10장)
(True, 0.584) [B급 프로그래머] 1월 5주 소식(빅데이터/인공지능 읽을거리 부문)
(True, 0.999) [운영체제(OS)] 9. 프로세스 동기화 2
(True, 1.000) 자바 String StringBuilder 그리고 StringBuffer 차이 비교
(True, 0.892) (업무)2020년 2월 3일 REACT로 임상시험자동화솔루션 개발 Start
(True, 0.544) [운영체제(OS)] 11. 모니터
(True, 0.999) [운영체제(OS)] 8. 프로세스 동기화 1
(True, 1.000) c언어 fopen 함수 : 파일을 연다.
(True, 0.726) 스프링 데이터 JPA와 Querydsl 인프런 강의 정리
(True, 0.991) [Sprint #10] Server Side Techniques Sprint
(True, 0.463) 기술 뉴스 #143 : 20-02-03
(True, 0.632) 카카오메일 IMAP / POP3 / SMTP 설정방법
(True, 0.985) [Algorithm] 이진트 리의 구현과 순회 알고리즘
(True, 0.999) [운영체제(OS)] 10. 프로세스 동기화 3
(True, 0.989) [맥북] 트랙패드 제스처
(True, 1.000) [스프링 부트 개념과 활용] 로깅
(True, 1.000) [스프링 부트 개념과 활용] Profile
```

## Data
- [전처리 및 라벨링 된 데이터](https://drive.google.com/drive/u/0/folders/1Npfrh6XmeABJ8JJ6ApS1T88vVoqyDH7M) : 23.5MB
- [wiki 한국어 데이터](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz) : 4.49GB 
- [Gensim FastText model](https://radimrehurek.com/gensim/models/fasttext.html) : 14.5GB
- tensorflow model : 170KB


## How To Run
### Environment
  - macOS Catalina v10.15.3
  - Python 3.7
  - Tensorflow 2.0.1

### Install library
  ~~~
  $ pip install -r requirements.txt
  ~~~

### Train
  ~~~
  $ python train.py
  ~~~

### Predict  
  ~~~
  $ python predict.py --predict '필리핀 세부 시티의 맛집! 하우스 오브 레촌 cebu city House of Lechon'
  > [{'text': '필리핀 세부 시티의 맛집! 하우스 오브 레촌 cebu city House of Lechon', 'predict': (False, 0.051)}]
  ~~~
  ~~~
  $ python predict.py --predict '쿠버네티스 CI/DI 를 위한 오픈소스 프로젝트 알아보기'
  > [{'text': '쿠버네티스 CI/DI 를 위한 오픈소스 프로젝트 알아보기', 'predict': (True, 0.989)}]
  ~~~
  ~~~
  $ python predict.py --predict '파이썬, 맛집탐방'
  > [{'text': '파이썬','predict': (True, 1.0)}, {'text': '맛집탐방', 'predict': (False, 0.073)}]
  ~~~
