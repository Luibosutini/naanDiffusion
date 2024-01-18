import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from transformers import AutoTokenizer
import streamlit as st

# デバイスの設定
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(418)

# 定数の定義
SEED = 418

# torchのデバイスを設定
torch_device = torch.device(device)

# 再現性のためにシードを設定
torch.manual_seed(SEED)

user_token = ""

# Streamlitアプリのメイン関数
def main():
    # ユーザーからの入力を受け取る
    user_token = st.text_input("ユーザートークンを入力してください", key="user_token")

    text_input = st.text_input("テキストを入力してください", key="text_input")

    # ((eating naan))を追加して文字列を更新
    text = text_input + " ((eating naan))"

    # 生成された画像を表示
    if st.button("画像を生成して表示"):
        generated_image = generate_image(text)
        st.image(generated_image, caption="Generated Image", use_column_width=True)

# StableDiffusionPipelineを使用して画像を生成
def generate_image(text):
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=user_token)
    model =  model.to(device)
    image = model(prompt=text, num_inference_steps = 20).images[0]
    return image

if __name__ == "__main__":
    main()