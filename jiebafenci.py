import jieba
import re


def segment(texts):
    results = []
    for text in texts:
        seg_list = jieba.cut(text)
        seg_text = ' '.join(seg_list)
        results.append(seg_text)
    return results

def is_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    match = re.search(pattern, text)
    return match is not None