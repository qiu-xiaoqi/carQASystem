import json
import numpy as np
from typing import List, Dict, Any
from text2vec import SentenceModel, semantic_search

class AnswerEvaluator:
    def __init__(self, sim_model_path: str = './pre_train_model/text2vec-base-chinese'):
        """初始化评估器"""
        self.sim_model = SentenceModel(model_name_or_path=sim_model_path, device='cuda:0')
    
    def calc_jaccard(self, list_a: List[str], list_b: List[str], threshold: float = 0.3) -> int:
        """计算Jaccard相似度并二值化"""
        size_c = len([i for i in list_a if i in list_b])
        score = size_c / (len(list_b) + 1e-6)
        return 1 if score > threshold else 0
    
    def evaluate_answer_pair(self, gold_answer: str, pred_answer: str, keywords: List[str]) -> float:
        """评估单个答案对"""
        if gold_answer == "无答案":
            return 1.0 if pred_answer == gold_answer else 0.0
        
        # 语义相似度计算
        gold_embedding = self.sim_model.encode([gold_answer])
        pred_embedding = self.sim_model.encode([pred_answer])
        semantic_score = semantic_search(gold_embedding, pred_embedding, top_k=1)[0][0]['score']
        
        # 关键词匹配
        matched_keywords = [word for word in keywords if word in pred_answer]
        keyword_score = self.calc_jaccard(matched_keywords, keywords)
        
        return 0.5 * keyword_score + 0.5 * semantic_score
    
    def generate_report(self, gold_path: str, pred_path: str) -> Dict[str, Any]:
        """生成评估报告"""
        with open(gold_path, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
        
        results = []
        for gold_item, pred_item in zip(gold_data, pred_data):
            question = gold_item["question"]
            score = self.evaluate_answer_pair(
                gold_item["answer"].strip(),
                pred_item["answer_4"].strip(),
                gold_item["keywords"]
            )
            
            result = {
                "question": question,
                "gold_answer": gold_item["answer"],
                "pred_answer": pred_item["answer_4"],
                "keywords": gold_item["keywords"],
                "score": score
            }
            results.append(result)
            print(f"问题: {question[:30]}... | 得分: {score:.4f}")
        
        return results
    
    @staticmethod
    def save_metrics(results: List[Dict], output_path: str) -> None:
        """保存评估结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 配置路径
    PATHS = {
        "gold": "./data/gold.json",
        "pred": "./data/result.json",
        "metrics": "./data/metrics.json"
    }
    
    print("="*80)
    print("开始评估...")
    print(f"标准答案路径: {PATHS['gold']}")
    print(f"预测结果路径: {PATHS['pred']}")
    
    # 初始化评估器
    evaluator = AnswerEvaluator()
    
    # 生成评估报告
    evaluation_results = evaluator.generate_report(PATHS['gold'], PATHS['pred'])
    
    # 计算平均分
    avg_score = np.mean([item["score"] for item in evaluation_results])
    
    # 输出总结
    print("\n" + "="*80)
    print(f"评估完成 | 总问题数: {len(evaluation_results)} | 平均得分: {avg_score:.4f}")
    print("="*80)
    
    # 保存结果
    evaluator.save_metrics(evaluation_results, PATHS['metrics'])
    print(f"\n评估结果已保存至: {PATHS['metrics']}")