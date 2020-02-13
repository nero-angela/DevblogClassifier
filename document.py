import requests, json
import os, re, csv
import pandas as pd
from enum import Enum
from tqdm import trange

class KEYS(Enum):
    ID = '_id'
    TITLE = 'title'
    DESC = 'description'
    TAGS = 'tags'
    LINK = 'link'
    
    def getList():
        return [k.value for k in KEYS]

class Document():
    
    def __init__(self, debug=False):
        self.debug = debug
        
        # Constant
        self.DATA_URL = 'https://awesome-devblog.now.sh/api/korean/people/feeds'
        self.DOCUMENTS_PATH = './data/documents.csv'
        self.MAX_REQ_SIZE = 5000
        
        if debug: print('> Updating documents...')
        self._updateDocs()
        
        if debug: print('> Loading documents...')
        self.docs = self._getDocs()
        
        if debug: print('> Done!')
        
    def _getTotal(self):
        """
        전체 문서 개수 요청
        """
        res = requests.get(self.DATA_URL, { 'size': 1 })
        res.raise_for_status()
        doc = res.json()
        return doc['total'][0]['count']

    def _reqDoc(self, page, size):
        """
        문서 요청
        - page는 0 부터 시작
        - 전처리(self._preprocessing) 후 반환
        """
        page += 1
        params = {
            'sort': 'date.asc',
            'page': page,
            'size': size
        }
        res = requests.get(self.DATA_URL, params)
        res.raise_for_status()
        doc = res.json()
        return self._preprocessing(doc)
    
    def _preprocessing(self, doc):
        """
        문서 전처리
        - KEYS 이외의 key 삭제
        - [tag] list join to string
        - [title / description / tags] 영어, 한글, 공백 이외의 것들 모두 삭제
        - \n, \t 삭제
        - 2번 이상의 공백 1개로 통합
        - 영어 대문자 소문자로 변환
        - 앞뒤 공백 삭제
        """
        for data in doc['data']:
            # KEYS 이외의 key 삭제
            rm_keys = data.keys() - KEYS.getList()
            for rm_key in rm_keys:
                del data[rm_key]
            
            # preprocessing
            for key in KEYS.getList():
                if key == 'tags':
                    data[key] = ' '.join(data[key])
                if key in ['title', 'description', 'tags']:
                    data[key] = re.sub('[^가-힣a-zA-Z\s]', '', data[key])
                data[key] = re.sub('\n', '', data[key])
                data[key] = re.sub('\t', '', data[key])
                data[key] = re.sub('\s{2,}', ' ', data[key])
                data[key] = data[key].lower()
                data[key] = data[key].strip()
        return doc
        

    def _reqDocs(self, size, start_page=0):
        """
        전체 문서 요청
        """
        total = self._getTotal()
        if size > self.MAX_REQ_SIZE: size = self.MAX_REQ_SIZE
        total_req = round(total/size + 0.5)
        docs = None
        for i in trange(start_page, total_req):
            doc = self._reqDoc(i, size)
            if docs == None:
                docs = doc
            else:
                docs['data'] += doc['data']
        return docs

    def _updateDocs(self):
        """
        최신 문서 추가
        - 데이터가 없는 경우, 전체 데이터를 가져옴
        - 기존 데이터가 있는 경우, 없는 데이터만 추가
        """
        size = self.MAX_REQ_SIZE
        
        if not os.path.isfile(self.DOCUMENTS_PATH):
            # 데이터가 없는 경우
            docs = self._reqDocs(size)
            with open(self.DOCUMENTS_PATH, 'w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                csv_writer.writerow(['_no', '_label'] + KEYS.getList())
                for i, doc in enumerate(docs['data']):
                    csv_writer.writerow([i + 1, -1] + [re.sub('\t', '', str(doc[k])) for k in KEYS.getList()])
        else:
            # 기존 데이터가 있는 경우
            num_new_docs = 0
            document = pd.read_csv(self.DOCUMENTS_PATH, delimiter=',')
            total = self._getTotal()
            old_total = document.tail(1)['_no'].values[0] # 기존 데이터 수
            new_docs_num = total - old_total
            if new_docs_num <= 0:
                if self.debug: print('The document is already up to date.')
                return
            
            docs = self._reqDocs(size, old_total // size)
            no = old_total + 1
            with open(self.DOCUMENTS_PATH, 'a') as tsv_file:
                csv_writer = csv.writer(tsv_file, delimiter=',')
                for i, doc in enumerate(docs['data']):
                    if doc['_id'] not in document._id.unique():
                        num_new_docs += 1
                        csv_writer.writerow([no + i, -1] + [re.sub('\t', '', str(doc[k])) for k in KEYS.getList()])
            if self.debug: print(f'신규 문서 {num_new_docs}개 추가')

    def _getDocs(self):
        """
        전체 문서 조회
        """
        if not os.path.isfile(self.DOCUMENTS_PATH):
            raise FileNotFoundError(f'The scripts file {self.DOCUMENTS_PATH} does not exist')
        return pd.read_csv(self.DOCUMENTS_PATH, delimiter=',')
    
    def syncDocLabel(self, old_document_path, delimiter):
        """
        기존 라벨링한 데이터를 신규 문서에 반영
        """
        def preprocessing(text):
            text = re.sub('\n', '', text)
            text = re.sub('\t', '', text)
            text = re.sub('\s{2,}', ' ', text)
            text = text.lower()
            text = text.strip()
            return text
        
        document = pd.read_csv(self.DOCUMENTS_PATH, delimiter=',')
        old_document = pd.read_csv(old_document_path, delimiter=delimiter)
        for index, row in old_document[:10000].iterrows():
            link = preprocessing(row.link)
            title = re.sub('[^가-힣a-zA-Z\s]', '', row.title)
            title = preprocessing(title)
            
            label = int(row.label)
            if not len(document.loc[document.title.str.strip() == title.strip()]) and not len(document.loc[document.link == link]):
                print(f'not found : {row.title}')
            elif len(document.loc[document.title.str.strip() == title.strip()]):
                document.loc[document.title.str.strip() == title.strip(), '_label'] = label
            elif len(document.loc[document.link == link]):
                document.loc[document.link == link, '_label'] = label
        
        # save synchronized document
        document.to_csv(self.DOCUMENTS_PATH, sep=",", index=False)
        return document
