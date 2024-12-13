import warnings
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')

params_count = {
    'analyzer': 'word',     # 取值'word'-分词结果为词级、'char'-字符级(结果会出现he is，空格在中间的情况)、'char_wb'-字符级(以单词为边界)，默认值为'word'
    'binary': False,        # boolean类型，设置为True，则所有非零计数都设置为1.（即，tf的值只有0和1，表示出现和不出现）
    'decode_error': 'strict',
    'dtype': np.float64,    # 输出矩阵的数值类型
    'encoding': 'utf-8',
    'input': 'content',     # 取值filename，文本内容所在的文件名；file，序列项必须有一个'read'方法，被调用来获取内存中的字节；content，直接输入文本字符串
    'lowercase': True,      # boolean类型，计算之前是否将所有字符转换为小写。
    'max_df': 1.0,          # 词汇表中忽略文档频率高于该值的词；取值在[0,1]之间的小数时表示文档频率的阈值，取值为整数时(>1)表示文档频数的阈值；如果设置了vocabulary，则忽略此参数。
    'min_df': 1,            # 词汇表中忽略文档频率低于该值的词；取值在[0,1]之间的小数时表示文档频率的阈值，取值为整数时(>1)表示文档频数的阈值；如果设置了vocabulary，则忽略此参数。
    'max_features': None,   # int或None(默认值).设置int值时建立一个词汇表，仅用词频排序的前max_features个词创建语料库；如果设置了vocabulary，则忽略此参数。
    'ngram_range': (1, 2),  # 要提取的n-grams中n值范围的下限和上限，min_n <= n <= max_n。
    'preprocessor': None,   # 覆盖预处理（字符串转换）阶段，同时保留标记化和 n-gram 生成步骤。仅适用于analyzer不可调用的情况。
    'stop_words': 'english',    # 仅适用于analyzer='word'。取值english，使用内置的英语停用词表；list，自行设置停停用词列表；默认值None，不会处理停用词
    'strip_accents': None,
    'token_pattern': '(?u)\\b\\w\\w+\\b',   # 分词方式、正则表达式，默认筛选长度>=2的字母和数字混合字符（标点符号被当作分隔符）。仅在analyzer='word'时使用。
    'tokenizer': None,      # 覆盖字符串标记化步骤，同时保留预处理和 n-gram 生成步骤。仅适用于analyzer='word'
    'vocabulary': None,     # 自行设置词汇表（可设置字典），如果没有给出，则从输入文件/文本中确定词汇表
}
params_tfidf = {
    'norm': 'l2',           # 输出结果是否标准化/归一化。l2：向量元素的平方和为1，当应用l2范数时，两个向量之间的余弦相似度是它们的点积；l1：向量元素的绝对值之和为1
    'smooth_idf': True,     # 在文档频率上加1来平滑 idf ，避免分母为0
    'sublinear_tf': True,  # 应用次线性 tf 缩放，即将 tf 替换为 1 + log(tf)
    'use_idf': True,        # 是否计算idf，布尔值，False时idf=1。
}


def text_embedding(data):
    '''
    class LemmaTokenizer:
        def __init__(self):
            self.wnl = WordNetLemmatizer()

        def __call__(self, doc):
            words = []
            for t in word_tokenize(doc):
                if len(t) < 3 or "'" in t or "~" in t:
                    continue
                words.append(self.wnl.lemmatize(t))
            return words

    params_count['tokenizer'] = LemmaTokenizer()
    '''

    params_count['max_features'] = 500
    params_count['max_df'] = 0.8
    params_count['min_df'] = 0.01

    cv = CountVectorizer(**params_count)
    x_cv = cv.fit_transform(data)

    vocabulary = cv.get_feature_names_out()
    print(vocabulary)
    # print(x_cv.toarray())

    tt = TfidfTransformer(**params_tfidf)
    x_tfidf = tt.fit_transform(x_cv.toarray())
    # print(x_tfidf.toarray())

    return x_tfidf.toarray()


if __name__ == "__main__":
    train_data = ["Chinese Beijing Chinese",
                  "Chinese Chinese Shanghai",
                  "Chinese Macao",
                  "Tokyo Japan Chinese"]
    text_embedding(train_data)
