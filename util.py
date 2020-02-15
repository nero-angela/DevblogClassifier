import urllib.request
from tqdm import tqdm

def downloadByURL(url, output_path):
    """
    HTTP 파일 다운로드
    """
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
        
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        
def oneHotEncoding(label, classNum):
    """
    라벨링된 int를 oneHot 인코딩한다
    ex) oneHotEncoding(0, 2) -> [1, 0]
    ex) oneHotEncoding(1, 2) -> [0, 1]
    """
    oneHot = [0]*classNum
    oneHot[label] = 1
    return oneHot
