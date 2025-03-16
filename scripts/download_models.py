import os
from transformers import LlamaTokenizer, LlamaModel, LlamaConfig
from huggingface_hub import snapshot_download

def download_llama():
    # 设置模型保存路径
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pretrained_models")
    model_path = os.path.join(cache_dir, "llama-7b")
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    print("Downloading LLaMA model and tokenizer...")
    try:
        # 下载完整的模型仓库
        snapshot_download(
            repo_id="huggyllama/llama-7b",
            cache_dir=cache_dir,
            local_dir=model_path
        )
        print(f"Successfully downloaded model to {model_path}")
        
        # 验证下载
        config = LlamaConfig.from_pretrained(model_path)
        model = LlamaModel.from_pretrained(model_path)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        print("Successfully loaded model and tokenizer")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

if __name__ == "__main__":
    download_llama() 