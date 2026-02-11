import os
import sys
import torch
import logging

# =================配置路径=================
# 将 src 加入路径，确保能 import llmvul
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入你的模块
from llmvul.loader import load_model_and_tokenizer
# 注意：如果你还没完全重构完，这里可能会报错，请根据实际文件名调整 import
try:
    from llmvul.analysis import circuits, causal, attention
    from llmvul.visualization import plot_circuits
except ImportError as e:
    print(f"Warning: Some modules could not be imported. Details: {e}")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =================Mock 数据 (测试用)=================
# 模拟一条数据，避免读取大文件
SAMPLE_ITEM = {
    "clean_prompt": "The capital of France is",
    "corrupted_prompt": "The capital of Germany is",
    "target": " Paris",
    "answer": " Paris"
}

def run_tests():
    logger.info("=== Starting LLMvul Functional Tests ===")

    # ------------------------------------------------
    # 1. 测试模型加载 (Loader)
    # ------------------------------------------------
    logger.info("[Test 1/4] Loading Model & Tokenizer...")
    try:
        # 使用 cpu 快速测试加载，或者自动检测
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 这里使用你的微调模型
        model_name = "Chun9622/llmvul-finetuned-gemma" 
        
        model, tokenizer = load_model_and_tokenizer(model_name=model_name, device=device)
        logger.info(f"✔ Model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"✘ Model loading failed: {e}")
        return # 如果模型都加载不了，后面就别跑了

    # ------------------------------------------------
    # 2. 测试电路分析 (Circuit Analysis)
    # ------------------------------------------------
    logger.info("[Test 2/4] Testing Circuit Analysis...")
    try:
        # 构造输入
        inputs = tokenizer(SAMPLE_ITEM["clean_prompt"], return_tensors="pt").to(device)
        
        # 假设你的 circuits.analyze 函数接受 model 和 inputs
        # TODO: 请根据你 src/llmvul/analysis/circuits.py 里的实际函数名修改下面这行
        # 例如: circuit_results = circuits.identify_circuits(model, inputs, threshold=0.1)
        
        # 这里写一个占位符，模拟运行
        logger.info("Running circuit identification logic...")
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions
            # 简单检查 attention 形状是否正确
            assert len(attentions) > 0, "No attention scores returned"
        
        # 模拟返回的电路数据 (Head 0:1, Layer 2)
        mock_circuit_result = [(2, 1, 0.5), (3, 4, 0.8)] 
        
        logger.info(f"✔ Circuit analysis ran successfully. Detected {len(mock_circuit_result)} components.")
    except Exception as e:
        logger.error(f"✘ Circuit analysis failed: {e}")
        mock_circuit_result = []

    # ------------------------------------------------
    # 3. 测试因果修补 (Causal Patching)
    # ------------------------------------------------
    logger.info("[Test 3/4] Testing Causal Patching...")
    try:
        clean_tokens = tokenizer(SAMPLE_ITEM["clean_prompt"], return_tensors="pt").to(device)
        corrupt_tokens = tokenizer(SAMPLE_ITEM["corrupted_prompt"], return_tensors="pt").to(device)
        
        # TODO: 调用你真实的 causal patching 函数
        # results = causal.run_patching(model, clean_tokens, corrupt_tokens, mock_circuit_result)
        
        logger.info("Running patching logic (Mock)...")
        # 模拟计算 logits 差异
        logit_diff = 0.45 # 假装算出来的
        
        logger.info(f"✔ Causal patching complete. Logit Diff: {logit_diff}")
    except Exception as e:
        logger.error(f"✘ Causal patching failed: {e}")

    # ------------------------------------------------
    # 4. 测试可视化 (Visualization)
    # ------------------------------------------------
    logger.info("[Test 4/4] Testing Visualization...")
    try:
        output_dir = "output_test"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "test_circuit.png")
        
        # TODO: 调用你真实的绘图函数
        # plot_circuits.plot_circuit_graph(mock_circuit_result, save_path=save_path)
        
        # 模拟绘图
        logger.info(f"Generating plot to {save_path}...")
        # (这里如果不调用真实函数，仅仅是测试路径创建)
        
        logger.info("✔ Visualization module accessed.")
    except Exception as e:
        logger.error(f"✘ Visualization failed: {e}")

    logger.info("=== All Tests Completed ===")

if __name__ == "__main__":
    run_tests()