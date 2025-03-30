"""
切分pdf文档
"""

import pdfplumber
from PyPDF2 import PdfReader


class DataProcess(object):

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.data = [] # 存储切分后的数据


    def SlidingWindow(self, sentences, kernel=512, stride=1):
        """
        滑动窗口切分pdf，主要是保证文本内容的跨页连续性问题
        结果类似于如下：
        句子1。句子2。句子3。
        句子2。句子3。句子4。
        句子3。句子4。句子5。
        """
        # 获取pdf有多少个句子
        sz = len(sentences)
        cur = ""  # 用于存储当前窗口内的句子。
        fast = 0 # 快指针，用于遍历sentences列表
        slow = 0 # 慢指针，用于标记当前窗口的起始位置
        while(fast < sz):
            sentence = sentences[fast] # 当前遍历到的句子
            # 检查窗口内的句子加上当前句子的总长度是否超过kernel，且是否已经存在与self.data列表中。
            if (len(cur + sentence) > kernel and (cur + sentence) not in self.data):
                # 如果条件满足，将将当前窗口内的句子加上当前句子添加到 self.data 列表中。
                self.data.append(cur + sentence + "。")
                # 更新 cur，移除窗口起始位置的句子。
                cur = cur[len(sentences[slow] + "。"): ]
                # 移动 slow 指针，更新窗口起始位置。
                slow = slow + 1
            
            # 更新当前窗口内容：
            cur = cur + sentence + "。"
            # 移动 fast 指针，继续遍历下一个句子。
            fast = fast + 1

    def Datafilter(self, line, header, pageid, max_seq=1024):
        """
        数据过滤, 根据当前的文档内容的item划分句子，然后根据max_seq划分文档块
        """
        sz = len(line)
        if sz < 6:
            return 
        
        if sz > max_seq:

            if "■" in line:
                sentences = line.split("■")
            elif "•" in line:
                sentences = line.split("•")
            elif "\t" in line:
                sentences = line.split("\t")
            else:
                sentences = line.split("。")
            
            for subsentence in sentences:
                subsentence = subsentence.replace("\n", "")

                if len(subsentence) < max_seq and len(subsentence) > 5:
                    subsentence = subsentence.replace(",", "").replace("\n","").replace("\t","")
                    if subsentence not in self.data:
                        self.data.append(subsentence)

    def GetHeader(self, page):
        """
        提取页头一级标题
        """
        try:
            """
            [{'text': 'MAX', 
            'x0': 172.23300000000003, 
            'x1': 188.18914020000003, 
            'top': 300.2211584000001, 
            'doctop': 209837.37315840137, 
            'bottom': 308.1912584000001, 
            'upright': True, 
            'height': 7.970100000000002, 
            'width': 15.956140199999993, 
            'direction': 'ltr'},
            
            ]
            """
            lines = page.extract_words()[::]
        except:
            return None
        
        if len(lines) > 0:
            for line in lines:

                if "目录" in line["text"] or ".........." in line["text"]:
                    return None
                if line["top"] < 20 and line["top"] > 17:
                    return line["text"]
            
            # 没找到目录，则返回第一个words
            return lines[0]["text"]
        return None
    def ParseBlock(self, max_seq=1024):
        """
        在每页中按块提取内容，并和一级标题进行组合
        """
        with pdfplumber.open(self.pdf_path) as pdf:
            for i , p in enumerate(pdf.pages):
                header = self.GetHeader(p)

                if header == None:
                    continue

                texts = p.extract_words(use_text_flow=True, extra_attrs = ["size"])[::]
                squence = ""
                last_size = 0

                # 按字体大小变化划分文本块
                for idx, line in enumerate(texts):
                    if idx < 1:
                        continue
                    if idx == 1:
                        if line["text"].isdigit():
                            continue

                    cursize = line["size"]
                    text = line["text"]

                    if text == "□" or text == "•":
                        continue
                    elif text== "警告！" or text == "注意！" or text == "说明！":
                        if len(squence) > 0:
                            self.Datafilter(squence, header, i, max_seq=max_seq)
                        squence = ""
                    elif format(last_size, ".5f") == format(cursize, ".5f"):
                        if len(squence) > 0:
                            squence = squence + text
                        else:
                            squence = text
                    else:
                        last_size = cursize
                        if len(squence) < 15 and len(squence) > 0:
                            squence = squence + text
                        else:
                            if len(squence) > 0:
                                self.Datafilter(squence, header, i, max_seq=max_seq)
                            squence = text

                # 最后处理剩余的文本块
                if len(squence) > 0:
                    self.Datafilter(squence, header, i, max_seq=max_seq)

    def ParseOnePageWithRule(self, max_seq=512, min_len=6):
        """
        按句号划分文档，利用最大长度划分文档块
        """
        for idx, page in enumerate(PdfReader(self.pdf_path).pages):
            page_content = ""
            text = page.extract_text()
            words = text.split('\n')
            for idx, word in enumerate(words):
                text = word.strip().strip("\n")
                if "...................." in text or "目录" in text:
                    continue

                if len(text) < 1:
                    continue
                if text.isdigit():
                    continue
                page_content = page_content + text
            
            if len(page_content) < min_len:
                continue
            if len(page_content) < max_seq:
                if page_content not in self.data:
                    self.data.append(page_content)
            else:
                sentences = page_content.split("。")
                cur = ""
                for idx, sentence in enumerate(sentences):
                    if len(cur + sentence) > max_seq and (cur + sentence) not in self.data:
                        self.data.append(cur + sentence)
                        cur = sentence
                    else:
                        cur = cur + sentence
        

    def ParseAllPage(self, max_seq=512, min_len=6):
        """
        提取所有页面内容
        """
        all_content = ""
        # 先读取pdf的页面内容
        for idx, page in enumerate(PdfReader(self.pdf_path).pages):
            page_content = ""
            text = page.extract_text()
            words = text.split('\n')

            # 去除空白和换行符得到连续的一页内容
            for idx, word in enumerate(words):
                text = word.strip().strip("\n")
                if ("...................." in text or "目录" in text):
                    continue
                if(len(text) < 1):
                    continue
                if(text.isdigit()):
                    continue
                page_content = page_content + text

            if(len(page_content) < min_len):
                continue
            all_content = all_content + page_content

        # 使用句号切分句子，使其成为数组，然后使用滑动窗口对数组进行切分
        sentences = all_content.split('。')
        self.SlidingWindow(sentences, kernel=max_seq)

if __name__ == '__main__':

    """
    项目最终采用了三种解析方案的综合：
    ● pdf分块解析，尽量保证一个小标题+对应文档在一个文档块，其中文档块的长度分别是512和1024。
    ● pdf滑窗法解析，把文档句号分割，然后构建滑动窗口，其中文档块的长度分别是256和512。
    ● pdf非滑窗法解析，把文档句号分割，然后按照文档块预设尺寸均匀切分，其中文档块的长度分别是256和512。
    按照3种解析方案对数据处理之后，然后对文档块做了一个去重，最后把这些文档块输入给召回模块。
    """

    pdf_path = 'data/train_a.pdf'
    dp = DataProcess(pdf_path)
    # 先按照块划分
    dp.ParseBlock(max_seq=1024)
    dp.ParseBlock(max_seq=512)
    print(len(dp.data))

    dp.ParseAllPage(max_seq=256)
    dp.ParseAllPage(max_seq=512)
    print(len(dp.data))

    dp.ParseOnePageWithRule(max_seq=256)
    dp.ParseOnePageWithRule(max_seq=512)
    print(len(dp.data))
    data = dp.data
    out = open("all_text.txt", "w", encoding="utf-8")
    for line in data:
        line = line.strip("\n")
        out.write(line)
        out.write("\n")
    out.close()
