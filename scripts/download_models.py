import os
from transformers import LlamaTokenizer, LlamaModel, LlamaConfig
from huggingface_hub import snapshot_download
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def setup_proxy():
    """设置代理和请求配置"""
    # 设置代理
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:1080"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1080"
    
    # 安装必要的依赖
    os.system("pip install requests[socks] PySocks")
    
    # 配置请求重试策略
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session

def download_llama():
    # 设置模型保存路径
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pretrained_models")
    model_path = os.path.join(cache_dir, "llama-7b")
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    print("Setting up proxy and dependencies...")
    session = setup_proxy()
    
    print("Downloading LLaMA model and tokenizer...")
    try:
        # 下载完整的模型仓库
        snapshot_download(
            repo_id="huggyllama/llama-7b",
            cache_dir=cache_dir,
            local_dir=model_path,
            local_files_only=False,
            token=os.getenv("HF_TOKEN"),  # 如果需要token
            proxies={
                "http": os.getenv("HTTP_PROXY", "http://127.0.0.1:1080"),
                "https": os.getenv("HTTPS_PROXY", "http://127.0.0.1:1080")
            }
        )
        print(f"Successfully downloaded model to {model_path}")
        
        # 验证下载
        print("Verifying downloaded files...")
        config = LlamaConfig.from_pretrained(model_path)
        model = LlamaModel.from_pretrained(model_path)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        print("Successfully loaded model and tokenizer")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify proxy settings")
        print("3. Make sure you have access to the model")
        print("4. Try setting HF_TOKEN environment variable")
        raise

if __name__ == "__main__":
    download_llama() 