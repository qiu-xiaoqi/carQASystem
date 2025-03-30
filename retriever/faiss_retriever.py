
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import os


class FaissRetriever(object):

    def __init__(self, model_path, data, vector_path=None):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path, 
            model_kwargs={"device":"cuda"},
            encode_kwargs={"batch_size": 64})
        self.vector_path = vector_path

        # 如果已有向量索引文件，则直接加载
        if os.path.exists(self.vector_path):
            print(f"加载已有Faiss索引从 {self.vector_path}")
            self.vector_store = FAISS.load_local(self.vector_path, self.embeddings)
        else:
            # 否则创建新索引
            print(f"创建新的Faiss索引并保存到 {self.vector_path}")
            docs = self._process_data(data)
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            self.vector_store.save_local(self.vector_path)
        
        del self.embeddings
        torch.cuda.empty_cache()

    def _process_data(self, data):
        docs = []
        for idx, line in enumerate(data):
            line = line.strip("\n").strip()
            words = line.split("\t")
            # 确保每个条目至少有内容
            page_content = words[0] if len(words) > 0 else ""
            metadata = {"id": idx}
            if len(words) > 1:
                metadata["text"] = words[1]
            docs.append(Document(page_content=page_content, metadata=metadata))
        return docs
    
    def update_index(self, new_data):
        """
        增量更新索引
        
        参数:
            new_data: 要添加的新数据列表
        """
        # 重新加载嵌入模型        
        # 处理新数据
        new_docs = self._process_data(new_data)
        new_vectors = self.embeddings.embed_documents([doc.page_content for doc in new_docs])
        
        # 添加到现有索引
        self.vector_store.add_embeddings(
            text_embeddings=zip([doc.page_content for doc in new_docs], new_vectors),
            metadatas=[doc.metadata for doc in new_docs]
        )
        
        # 保存更新后的索引
        self.vector_store.save_local(self.vector_path)
        
        # 清理资源
        del self.embeddings
        torch.cuda.empty_cache()

    def getTopK(self, query, k):
        context = self.vector_store.similarity_search_with_score(query, k=k)
        return context

    def getVectorStore(self):
        self.vector_store = FAISS.load_local("faiss_index", self.embeddings)
        return self.vector_store
    

    
if __name__ == "__main__":
    # 读取text文件
    text_path = "all_text.txt"
    with open(text_path, "r", encoding="utf-8") as f:
        data_list = f.readlines()
    
    model_path = "pre_train_model/m3e-large"
    vector_path = "faiss_index"
    
    retriever = FaissRetriever(
        model_path=model_path,
        data=data_list,
        vector_path=vector_path  # 可选
        )

    # 查询
    results = retriever.getTopK("如何预防新冠肺炎", k=3)
    print(results)

    # # 更新索引
    # retriever.update_index(new_data_list)