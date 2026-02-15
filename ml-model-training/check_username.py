from huggingface_hub import HfApi
import os

# Get token from environment variable
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("Please set HF_TOKEN environment variable")

api = HfApi(token=HF_TOKEN)
user_info = api.whoami()
print(f"Username: {user_info['name']}")
print(f"Full name: {user_info.get('fullname', 'N/A')}")
print(f"Type: {user_info['type']}")
