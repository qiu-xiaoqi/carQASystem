from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
import jieba

class BM25(object):

    def __init__(self, documents):

        docs = []
        full_docs = []
        for idx, line in enumerate(documents):
            line = line.strip("\n").strip()
            if len(line) < 5:
                continue
            tokens = " ".join(jieba.cut_for_search(line))
            docs.append(Document(page_content=tokens, 
                                 metadata={"id": idx}))
            words = line.split("\t")
            full_docs.append(Document(page_content=words[0],
                                      metadata={"id": idx}))
        self.documents = docs
        self.full_documents = full_docs 
        self.retriever = self._init_bm25()

    def _init_bm25(self):
        return BM25Retriever.from_documents(self.documents)

    def getBM25TopK(self, query, topk=5):
        self.retriever.k = topk
        query = " ".join(jieba.cut_for_search(query))
        ans_docs = self.retriever.get_relevant_documents(query)
        ans = []
        for line in ans_docs:
            ans.append(self.full_documents[line.metadata["id"]])
        return ans
    

if __name__ == "__main__":
    text_path = "all_text.txt"
    with open(text_path, "r", encoding="utf-8") as f:
        data_list = f.readlines()
    
    bm25 = BM25(data_list)
    res = bm25.getBM25TopK("座椅加热", topk=6)
    print(res)
