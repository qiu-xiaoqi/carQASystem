import json
import time
from vllm_model import Qwen7BChatModel
from rerank.rerank_model import reRankLLM
from retriever.chroma_retriever import ChromaRetriever
from retriever.bm25_retriever import BM25

def get_emb_bm25_merge(faiss_context, bm25_context, query):
    """合并FAISS和BM25召回结果构造prompt"""
    max_length = 2500
    emb_ans = ""
    for doc, score in faiss_context[:6]:
        if len(emb_ans + doc.page_content) > max_length:
            break
        emb_ans += doc.page_content
    
    bm25_ans = ""
    for doc in bm25_context[:6]:
        if len(bm25_ans + doc.page_content) > max_length:
            break
        bm25_ans += doc.page_content
    
    return f"""基于以下已知信息，简洁和专业的来回答用户的问题。
            如果无法从中得到答案，请说"无答案"，不允许在答案中添加编造成分。
            已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
            1: {emb_ans}
            2: {bm25_ans}
            问题: {query}"""

def get_rerank(context, query):
    """构造重排序后的prompt"""
    return f"""基于以下已知信息，简洁和专业的来回答用户的问题。
            如果无法从中得到答案，请说"无答案"，不允许在答案中添加编造成分。
            已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
            1: {context}
            问题: {query}"""

def reRank(rerank, top_k, query, bm25_ans, faiss_ans):
    """重排序并合并结果"""
    max_length = 4000
    items = [doc for doc, score in faiss_ans] + bm25_ans
    rerank_ans = rerank.predict(query, items)[:top_k]
    return "".join(doc.page_content for doc in rerank_ans[:6] 
                  if len(doc.page_content) <= max_length)

def load_data(text_path):
    """加载文本数据"""
    with open(text_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    start = time.time()
    base = "."
    
    # 初始化模型
    print("Initializing models...")
    chromaretriever = ChromaRetriever(
        model_path=f"{base}/pre_train_model/m3e-large",
        data=load_data("all_text.txt"),
        vector_path="chroma_db"
    )
    bm25 = BM25(load_data("all_text.txt"))
    llm = Qwen7BChatModel(
        f"{base}/pre_train_model/Qwen-7B-Chat",
        tensor_parallel_size=2
    )
    rerank = reRankLLM(f"{base}/pre_train_model/bge-reranker-large")
    
    # 加载测试问题
    with open(f"{base}/data/test_question.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    # 处理每个问题
    for item in test_data:
        query = item["question"]
        
        # 获取召回结果
        chroma_context = chromaretriever.getTopK(query, k=15)
        bm25_context = bm25.getBM25TopK(query, topk=15)
        
        # 准备各种prompt
        batch_inputs = [
            get_emb_bm25_merge(chroma_context, bm25_context, query),  # 合并召回
            get_rerank(
                "".join(doc.page_content for doc in bm25_context[:6]), 
                query
            ),  # BM25
            get_rerank(
                "".join(doc.page_content for doc, _ in chroma_context[:6]), 
                query
            ),  # 向量召回
            get_rerank(
                reRank(rerank, 6, query, bm25_context, chroma_context), 
                query
            )  # 重排序
        ]
        
        # 批量推理
        batch_output = llm.batch_infer(batch_inputs)
        
        # 保存结果
        item.update({
            "answer_1": batch_output[0],  # 合并召回
            "answer_2": batch_output[1],  # BM25
            "answer_3": batch_output[2],  # 向量召回
            "answer_4": batch_output[3],  # 重排序
            "chroma_score": chroma_context[0][1] if chroma_context else 0
        })
    
    # 保存结果
    with open(f"{base}/data/result.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"Total time: {(time.time()-start)/60:.2f} minutes")

if __name__ == "__main__":
    main()