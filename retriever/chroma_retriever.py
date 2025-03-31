from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import os


class ChromaRetriever(object):
    def __init__(self, model_path, data, vector_path=None, collection_name="my_collection"):
        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"batch_size": 64}
        )
        self.vector_path = vector_path
        self.collection_name = collection_name

        # 如果已有向量数据库，则直接加载
        if os.path.exists(self.vector_path):
            print(f"从 {self.vector_path} 加载已有 Chroma 索引")
            self.vector_store = Chroma(
                persist_directory=self.vector_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
        else:
            # 否则创建新索引
            print(f"创建新的 Chroma 索引并保存到 {self.vector_path}")
            docs = self._process_data(data)
            self.vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=self.vector_path,
                collection_name=self.collection_name
            )
        
        # 清理资源
        del self.embeddings
        torch.cuda.empty_cache()

    def _process_data(self, data):
        docs = []
        for idx, line in enumerate(data):
            line = line.strip("\n").strip()
            words = line.split("\t")
            # 确保每个条目至少有内容
            page_content = words[0] if len(words) > 0 else ""
            metadata = {"id": str(idx)}  # Chroma 建议使用字符串ID
            if len(words) > 1:
                metadata["text"] = words[1]
            docs.append(Document(page_content=page_content, metadata=metadata))
        return docs
    
    def update_index(self, new_data):
        """增量更新索引"""
        # 重新加载嵌入模型
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"batch_size": 64}
        )
        
        # 处理新数据
        new_docs = self._process_data(new_data)
        
        # 添加到现有集合
        self.vector_store.add_documents(
            documents=new_docs,
            embedding=embeddings
        )
        
        # 清理资源
        del embeddings
        torch.cuda.empty_cache()

    def getTopK(self, query, k=3):
        """获取相似度最高的k个结果"""
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def get_vector_store(self):
        """获取向量存储对象"""
        return self.vector_store


if __name__ == "__main__":
    # 读取text文件
    text_path = "all_text.txt"
    with open(text_path, "r", encoding="utf-8") as f:
        data_list = f.readlines()
    
    model_path = "pre_train_model/m3e-large"
    vector_path = "chroma_db"  # Chroma使用目录而不是文件
    
    retriever = ChromaRetriever(
        model_path=model_path,
        data=data_list,
        vector_path=vector_path
    )

    # 查询示例
    results = retriever.getTopK("座椅加热", k=3)
    for doc, score in results:
        print(f"内容: {doc.page_content}")
        print(f"元数据: {doc.metadata}")
        print(f"相似度得分: {score:.4f}")
        print("-" * 50)