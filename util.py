import urllib.request
import numpy as np
from tqdm import tqdm

def downloadByURL(url, output_path):
    """
    HTTP 파일 다운로드
    
    - input
    : url / str / 다운로드 받으려는 파일의 url
    : output_path / str / 파일 저장 경로
    """
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
        
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
