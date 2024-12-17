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
    """
    将输入的文本数据转换为文本嵌入向量，使用词袋模型和 TF-IDF 技术。

    Args:
        data: 输入文本数据，可以是字符串列表或其他可迭代对象。

    Returns:
        文本嵌入向量，以 NumPy 数组的形式返回。
    """

    # 定义一个用于文本预处理的类，继承自 CountVectorizer 的 tokenizer 参数
    class LemmaTokenizer:
        """
        用于文本预处理的类，进行词形还原和简单的过滤。
        """
        def __init__(self):
            # 初始化 WordNetLemmatizer 对象，用于词形还原
            self.wnl = WordNetLemmatizer()

        def __call__(self, doc):
            words = []  # 初始化一个空列表，用于存储预处理后的单词
            for t in word_tokenize(doc):  # 使用 word_tokenize 对文本进行分词
                # 过滤掉长度小于 3、包含撇号或波浪号的单词
                if len(t) < 3 or "'" in t or "~" in t:
                    continue
                # 对剩余的单词进行词形还原，并将结果添加到 words 列表中
                words.append(self.wnl.lemmatize(t))
            return words  # 返回预处理后的单词列表

    # 将 LemmaTokenizer 对象赋值给 params_count['tokenizer']，
    # 使 CountVectorizer 在创建词袋模型时使用 LemmaTokenizer 来对文本进行预处理。
    params_count['tokenizer'] = LemmaTokenizer()

    # 设置 CountVectorizer 的参数
    params_count['max_features'] = 500  # 限制特征词的数量为 500
    params_count['max_df'] = 0.8  # 忽略文档频率高于 0.8 的词
    params_count['min_df'] = 0.01  # 忽略文档频率低于 0.01 的词
    params_count['ngram_range'] = (1,1) #提取unigrams

    # 创建 CountVectorizer 对象，并使用 params_count 中的参数进行配置
    cv = CountVectorizer(**params_count)
    # 使用 CountVectorizer 拟合数据并转换文本
    x_cv = cv.fit_transform(data)

    # 获取特征词列表
    vocabulary = cv.get_feature_names_out()
    # 打印特征词列表
    print(f"特征词列表：\n{vocabulary}")
    # 打印词频矩阵
    print(f"词频矩阵：\n{x_cv.toarray()}")
    

    # 创建 TfidfTransformer 对象，并使用 params_tfidf 中的参数进行配置
    tt = TfidfTransformer(**params_tfidf)
    # 使用 TfidfTransformer 拟合数据并转换文本
    x_tfidf = tt.fit_transform(x_cv.toarray())
    # 打印 TF-IDF 矩阵
    print(f"TF-IDF 矩阵：\n{x_tfidf.toarray()}")

    # 返回 TF-IDF 矩阵作为文本嵌入向量
    return x_tfidf.toarray()

if __name__ == "__main__":
    train_data = ["Chinese Beijing Chinese ",
                  "Chinese Chinese Shanghai",
                  "Chinese Macao",
                  "Tokyo Japan Chinese"]
    text_embedding(train_data)
