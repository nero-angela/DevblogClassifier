import os, re, csv, requests, json
import numpy as np
import pandas as pd
from flags import CONST
from enum import Enum
from tqdm import trange
from bs4 import BeautifulSoup
from util import downloadByURL, downloadIfNotExist

class KEYS(Enum):
    # -1 : 아직 라벨링 안함 (default)
    # 0  : 개발과 관련없는 문서
    # 1  : 개발과 관련있는 문서
    LABEL = 'label'
    
    # TAGS + TITLE + DESC
    TEXT = 'text'
    
    # DATA_URL 결과 파싱용 Keys(Beans)
    ID = '_id'
    TITLE = 'title'
    DESC = 'description'
    TAGS = 'tags'
    LINK = 'link'
    
    def getDocKeys():
        """
        awesome-devblog API 요청시 가져오려는 컬럼
        
        - return
        : list / 컬럼명 리스트
        """
        return [KEYS.ID.value, KEYS.TITLE.value, KEYS.DESC.value, KEYS.TAGS.value, KEYS.LINK.value]
    
    def getTitleBlackList():
        """
        title 컬럼 기준 블랙리스트
        
        - return
        : list / 블랙리스트
        """
        return ['', 'about']
    
    def getTextKeys():
        """
        text 컬럼에 사용되는 awesome-devblog 컬럼
        
        - return
        : list / 컬럼명 리스트
        """
        return [KEYS.TAGS.value, KEYS.TITLE.value, KEYS.DESC.value]

class Document():
    
    def __init__(self, update=False):

        if update:
            self.updateDocs()
        
    def _getTotal(self):
        """
        awesome-devblog에 전체 문서 개수 요청
        
        - return
        : int / 전체 문서 개수
        """
        res = requests.get(CONST.origin_data_url, { 'size': 1 })
        res.raise_for_status()
        doc = res.json()
        return doc['total'][0]['count']

    def _reqDoc(self, page, size, preprocessing=False):
        """
        awesome-devblog에 문서 요청
        : KEYS에 지정된 컬럼만 가져옴
        
        - input
        : page / int / 요청 페이지(0부터 시작)
        : size / int / 한 번의 요청으로 가져오려는 문서 개수
        : preprocessing / boolean / 문서 전처리 여부
        
        - output
        : DataFrame / DataFrame(response['data'])
        """
        page += 1
        params = {
            'sort': 'date.asc',
            'page': page,
            'size': size
        }
        res = requests.get(CONST.origin_data_url, params)
        res.raise_for_status()
        doc = res.json()
        
        # json to dataframe
        doc = pd.DataFrame(doc['data'], columns=KEYS.getDocKeys())
        
        # add label
        doc.insert(0, KEYS.LABEL.value, -1)
        
        if preprocessing:
            return self.preprocessing(doc)
        else:
            return doc

    def _reqDocs(self, size, start_page=0):
        """
        awesome-devblog에 전체 문서 요청
        - input
        : size / int / 한 번의 요청으로 가져올 문서개수(max 5000)
        : start_page / int / 해당 페이지 부터 마지막 페이지까지 조회
        
        - return
        : DataFrame / 전처리된 전체 데이터로 구성
        """
        total = self._getTotal()
        if size > CONST.origin_max_req_size: size = CONST.origin_max_req_size
        total_req = round(total/size + 0.5)
        docs = pd.DataFrame()
        for i in trange(start_page, total_req):
            doc = self._reqDoc(i, size)
            if docs.empty:
                docs = doc
            else:
                docs = docs.append(doc)
        return self.preprocessing(docs)
    
    def preprocessing(self, doc, joinTags=True, devblog=False):
        r"""
        문서 전처리
        : tags / 배열로 되어있으므로 띄어쓰기로 join
        : title, description, tags / 영어, 한글, 공백만 남김
        : html tag 삭제
        : \n, \r 삭제
        : 2회 이상의 공백은 하나로 줄입
        : 영어 대문자 소문자로 변환
        : 앞뒤 공백 삭제
        : 블랙리스트 데이터(KEYS.getTitleBlackList()) 제외
        : text / tags + title + description 순서로 join된 컬럼 생성
        
        - input
        : doc / DataFrame or str / documents.csv DataFrame
        : joinTags / boolean / tags join 여부
        
        - return
        : DataFrame / 전처리 완료된 데이터
        """

        if type(doc) == str:
            doc = pd.DataFrame([{
                'title': doc,
                'description': '',
                'tags': []
            }])
        
        # title, description, tags
        def textPreprocessing(x):
            x = BeautifulSoup(str(x), "html.parser").get_text()
            if devblog:
                x = re.sub('[^ㄱ-ㅎㅏ-ㅣ_가-힣a-zA-Z\s]', '', x)
            else:
                x = re.sub('[^가-힣a-zA-Z\s]', '', x)
            return x
        
        # all
        def docPreprocessing(x):
            x = re.sub('[\n\r]', '', x)
            x = re.sub('\s{2,}', ' ', x)
            x = x.lower()
            x = x.strip()
            return x
        
        for key in doc.columns:
            if joinTags and KEYS(key) == KEYS.TAGS:
                doc[key] = doc[key].apply(lambda x: ' '.join(x))
            if key in KEYS.getTextKeys():
                doc[key] = doc[key].apply(textPreprocessing)
                
            if key in KEYS.getDocKeys():
                doc[key] = doc[key].apply(docPreprocessing)
            
        # remove blacklist
        doc = doc.drop(doc[doc[KEYS.TITLE.value].isin(KEYS.getTitleBlackList())].index).reset_index()
                        
        # create text column
        join_with = lambda x: ' '.join(x.dropna().astype(str)).strip()
        doc[KEYS.TEXT.value] = doc[KEYS.getTextKeys()].apply(
            join_with,
            axis=1
        )
        return doc
    
    def getDocs(self, labeled_only=True):
        """
        전체 문서 조회
        - input
        : labeled_only / boolean / 라벨링 된 데이터만 가져올지 선택
        
        - return
        : DataFrame / documents.csv 데이터
        """
        downloadIfNotExist(CONST.devblog_data_path, CONST.devblog_data_url)
        data = pd.read_csv(CONST.devblog_data_path, delimiter=',', dtype={KEYS.LABEL.value: np.int64})
        if not labeled_only:
            return data
        else:
            return data.loc[data.label != -1]
    
    def updateDocs(self):
        """
        awesome-devblog에 최신 문서 요청 및 documents.csv에 추가
        : 데이터가 없는 경우, 전체 데이터를 가져옴
        : 기존 데이터가 있는 경우, 없는 데이터만 추가
       
        - export
        : ./data/documents.csv가 없는 경우 신규 생성
        : ./data/documents.csv가 있는 경우 신규 문서 추가
        """
        size = CONST.origin_max_req_size
        
        if not os.path.isfile(CONST.devblog_data_path):
            # 데이터가 없는 경우
            docs = self._reqDocs(size)
            docs.to_csv(CONST.devblog_data_path, sep=",", index=False)
        else:
            # 기존 데이터가 있는 경우
            docs = pd.read_csv(CONST.devblog_data_path, delimiter=',')
            total = self._getTotal()
            total_docs = len(docs)
            new_docs = self._reqDocs(size, total_docs // size)
            
            # _id가 기존 데이터에 존재하지 않는 경우에만 추가
            docs = docs.append(new_docs[~new_docs[KEYS.ID.value].isin(docs[KEYS.ID.value])])
            docs.to_csv(CONST.devblog_data_path, sep=",", index=False)
            
            if total_docs == len(docs):
                print('🐈 문서가 최신 상태입니다.')
            else:
                print(f'🐈 신규 문서 {len(docs) - total_docs}개 추가')
    
    def syncDocLabel(self, old_document_path, sep, override=False):
        """
        기존 라벨링한 데이터를 신규 문서에 반영
        : title, link 기준으로 일치하는 문서 검색
        
        - input
        : old_document_path / str / 기존 라벨링한 데이터 경로
        : sep / str / csv delimiter
        : override / boolean / 기존 라벨링이 반영된 결과를 ./data/documents.csv로 저장여부
        
        - export
        : ./data/documents.csv
        """
        
        document = pd.read_csv(CONST.devblog_data_path, delimiter=',')
        old_document = pd.read_csv(old_document_path, delimiter=sep)
        self.preprocessing(old_document, joinTags=False)
        for index, row in old_document.iterrows():
            link = row.link
            title = row.title
            label = int(row.label)
            if not len(document.loc[document.title.str.strip() == title.strip()]) and not len(document.loc[document.link == link]):
                print(f'🐈 not found : {row.title}')
            elif len(document.loc[document.title.str.strip() == title.strip()]):
                document.loc[document.title.str.strip() == title.strip(), KEYS.LABEL.value] = label
            elif len(document.loc[document.link == link]):
                document.loc[document.link == link, KEYS.LABEL.value] = label
        
        # save synchronized document
        if override:
            document.to_csv(CONST.devblog_data_path, sep=",", index=False)
        print('🐈 done')
