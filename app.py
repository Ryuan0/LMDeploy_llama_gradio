import gradio as gr
from lmdeploy import pipeline, TurbomindEngineConfig
#####
import os

# download the model llava-v1.6-vicuna-7b to the base_path directory using git tool
base_path = './llava-v1.6-vicuna-7b'
os.system(f'git clone https://code.openxlab.org.cn/RYUAN0/llava-v1.6-vicuna-7b.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')
######

backend_config = TurbomindEngineConfig(session_len=8192) # 图片分辨率较高时请调高session_len
### pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config) 非开发机运行此命令
# pipe = pipeline('/share/new_models/liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config)

#### need to change the model path;
pipe = pipeline(base_path, backend_config=backend_config)

def model(image, text):
    if image is None:
        return [(text, "请上传一张图片。")]
    else:
        response = pipe((text, image)).text
        return [(text, response)]

demo = gr.Interface(fn=model, inputs=[gr.Image(type="pil"), gr.Textbox()], outputs=gr.Chatbot())
demo.launch()   
