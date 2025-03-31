import os
import torch
import time
import asyncio
import multiprocessing as mp
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from transformers import AutoTokenizer, GenerationConfig

# 多进程设置（必须放在最顶部）
mp.set_start_method('spawn', force=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 确保GPU顺序一致

class Qwen7BChatModel:
    def __init__(self, model_path, tensor_parallel_size=2):
        """初始化多GPU LLM引擎"""
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            pad_token='<|extra_0|>',
            eos_token='<|endoftext|>',
            padding_side='left',
            trust_remote_code=True
        )
        
        # 配置生成参数
        self.generation_config = GenerationConfig.from_pretrained(model_path)
        
        # 初始化同步LLM引擎
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            dtype="auto",
            max_model_len=8192,
        )
        
        # 默认采样参数
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
            presence_penalty=0.1,
            frequency_penalty=0.1,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )

    def build_chat_prompt(self, messages):
        """构建Qwen对话提示"""
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += f"<|im_start|>system\n{message['content']}<|im_end|>\n"
            elif message["role"] == "user":
                prompt += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
            elif message["role"] == "assistant":
                prompt += f"<|im_start|>assistant\n{message['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def generate_sync(self, prompts, sampling_params=None):
        """同步批量推理"""
        params = sampling_params or self.sampling_params
        outputs = self.llm.generate(prompts, params)
        return [output.outputs[0].text for output in outputs]

    async def generate_async(self, prompts, sampling_params=None):
        """异步批量推理"""
        params = sampling_params or self.sampling_params
        
        # 创建异步引擎
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            tokenizer=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            dtype="auto",
            max_model_len=8192,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # 生成任务
        tasks = [engine.generate(prompt, params) for prompt in prompts]
        
        # 等待所有任务完成
        outputs = await asyncio.gather(*tasks)
        return [output.outputs[0].text for output in outputs]

    def batch_infer(self, queries, system_prompt=None):
        """批量推理便捷方法"""
        system = system_prompt or "你是一个有帮助的AI助手。"
        prompts = [
            self.build_chat_prompt([
                {"role": "system", "content": system},
                {"role": "user", "content": query}
            ])
            for query in queries
        ]
        return self.generate_sync(prompts)

    def release(self):
        """显式释放资源"""
        if hasattr(self, 'llm'):
            del self.llm
        torch.cuda.empty_cache()

# 示例使用
if __name__ == "__main__":
    # 配置参数
    model_path = "/root/autodl-tmp/carQASystem/pre_train_model/Qwen-7B-Chat"
    tensor_parallel_size = 2  # 使用2张GPU卡
    
    # 初始化模型
    print(f"Initializing Qwen-7B-Chat on {tensor_parallel_size} GPUs...")
    start_time = time.time()
    model = Qwen7BChatModel(model_path, tensor_parallel_size)
    print(f"Model loaded in {time.time()-start_time:.2f}s")

    # 测试1: 同步批量推理
    print("\n=== 测试同步批量推理 ===")
    test_queries = [
        "解释吉利汽车的座椅按摩功能",
        "如何唤醒吉利汽车的语音助手？",
        "详细说明自动驾驶功能的技术原理",
    ]
    
    start_time = time.time()
    responses = model.batch_infer(test_queries)
    for query, response in zip(test_queries, responses):
        print(f"\nQ: {query}\nA: {response[:200]}...")
    print(f"\n批量推理耗时: {time.time()-start_time:.2f}s")

    # # 测试2: 异步推理
    # print("\n=== 测试异步推理 ===")
    # async def test_async():
    #     messages = [
    #         {"role": "system", "content": "你是一个汽车专家助手。"},
    #         {"role": "user", "content": "吉利汽车的智能座舱有哪些特点？"}
    #     ]
    #     prompt = model.build_chat_prompt(messages)
    #     responses = await model.generate_async([prompt])
    #     print(responses[0])
    
    # asyncio.run(test_async())

    # 释放资源
    model.release()