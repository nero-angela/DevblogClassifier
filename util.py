import re, os
import urllib.request
import numpy as np
from tqdm import tqdm

def downloadIfNotExist(path, url):
    if not os.path.isfile(path):
        print(f'🐈  {path} 파일을 다운로드 합니다.')
        downloadByURL(url, path)

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

def han2Jamo(str):
    INITIALS = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
    MEDIALS = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
    FINALS = list("_ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
    SPACE_TOKEN = " "
    LABELS = sorted({SPACE_TOKEN}.union(INITIALS).union(MEDIALS).union(FINALS))

    def check_hangle(char):
        return 0xAC00 <= ord(char) <= 0xD7A3

    def jamo_split(char):
        assert check_hangle(char)
        diff = ord(char) - 0xAC00
        _m = diff % 28
        _d = (diff - _m) // 28
        return (INITIALS[_d // 21], MEDIALS[_d % 21], FINALS[_m])
    
    result = ""
    for char in re.sub("\\s+", SPACE_TOKEN, str.strip()):
        if char == SPACE_TOKEN:
            result += SPACE_TOKEN
        elif check_hangle(char):
            result += "".join(jamo_split(char))
        else:
            result += char
    return result
