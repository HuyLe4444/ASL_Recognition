import torch
import textwrap
import cv2
import os

import numpy as np
import google.generativeai as genai
import torch.nn as nn
import matplotlib.pyplot as plt
import PIL.Image

from IPython.display import display
from IPython.display import Markdown

from dotenv import load_dotenv

from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

import time
from tqdm.auto import tqdm

# Initialize with numbers 0-9 for the first 10 characters
CHARACTER_MAPPING = {i: str(i) for i in range(10)}

# Add A-S for the next 19 characters
CHARACTER_MAPPING.update({10 + i: chr(65 + i) for i in range(19)})

# Add space as the 30th character
CHARACTER_MAPPING[29] = ' '

# Add the remaining letters T-Z
CHARACTER_MAPPING.update({30 + i: chr(84 + i) for i in range(7)})

# Verify the mapping
for i in range(37):
    print(f"{i}: {CHARACTER_MAPPING[i]}")
    
prompt = (
        "Describe this picture in detail, including the background, characters, actions, emotions, and any other notable details. "
        "Provide the output in the following JSON format:\n"
        "{\n"
        "  \"background\": \"\",\n"
        "  \"characters\": [\n"
        "    {\n"
        "      \"name\": \"\",\n"
        "      \"description\": \"\",\n"
        "      \"actions\": \"\",\n"
        "      \"emotions\": \"\",\n"
        "      \"clothing\": \"\"\n"
        "    }\n"
        "  ],\n"
        "  \"actions\": \"\",\n"
        "  \"emotions\": \"\",\n"
        "  \"significant_details\": \"\",\n"
        "  \"setting_significance\": \"\",\n"
        "  \"scene_connection\": \"\"\n"
        "}\n\n"
    )

class ASLModelV0(nn.Module):
  def __init__(self, input_feature, output_feature, hidden_unit = 32):
    super().__init__()
    self.layer1 = nn.Sequential(
        nn.Conv2d(input_feature, hidden_unit, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.Conv2d(hidden_unit, hidden_unit, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )
    self.layer2 = nn.Sequential(
        nn.Conv2d(hidden_unit, hidden_unit, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.Conv2d(hidden_unit, hidden_unit, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )

    self.layer3 = nn.Sequential(
        nn.Conv2d(hidden_unit, hidden_unit, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.Conv2d(hidden_unit, hidden_unit, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )

    self.layer4 = nn.Sequential(
        nn.Conv2d(hidden_unit, hidden_unit, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.Conv2d(hidden_unit, hidden_unit, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )


    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(hidden_unit * 14 * 14, output_feature)
    )
  def forward(self, x: torch.Tensor):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.classifier(x)
    return x

def load_model(model_path, device):
    model = ASLModelV0(input_feature=3, output_feature=37, hidden_unit=32)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    # model.eval()
    return model

def process_frames(model, frame_dir, device):
    result_string = ""
    
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    frame_count = 0
    while True:
        frame_name = f"frame_{frame_count}.jpg"
        frame_path = os.path.join(frame_dir, frame_name)
        if not os.path.exists(frame_path):
            break
        
        image = PIL.Image.open(frame_path)
        image_tensor = data_transform(image).unsqueeze(0).to(device)
        
        with torch.inference_mode():
            output = model(image_tensor)
            _, pred = torch.max(output, 1)
            char_index = pred.item()
            predicted_char = CHARACTER_MAPPING.get(char_index, '')
            result_string += predicted_char
        
        print(f"{frame_name}: Predicted Index: {char_index}, Character: {predicted_char}")
        
        frame_count += 1
    
    return result_string



def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def frame_cut(output_dir, video_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    frame_interval = int(fps * 1)
    frame_count = 0
    count = 0
    success = True

    while success:
        success, image = vidcap.read()
        if success and frame_count % frame_interval == 0:
            frame_time = frame_count / fps
            output_file = os.path.join(output_dir, f"frame_{count}.jpg")
            cv2.imwrite(output_file, image)
            count += 1
        frame_count += 1

    vidcap.release()
    print("Cắt ảnh từ video hoàn tất!")
    
def api_call_txt(output_dir):
    count = 0

    for img in os.listdir(output_dir):
        count += 1

    model = genai.GenerativeModel('gemini-1.5-flash')
    output_file = 'descriptions.text'

    with open(output_file, 'a') as output:
        # Loop qua từng frame từ 0 đến 50
        for i in tqdm(range(count)):
            file_name = f'frame_{i}.jpg'
            file_path = os.path.join(output_dir, file_name)
            
            # Kiểm tra nếu file tồn tại
            if os.path.exists(file_path):
                with open(file_path, 'rb') as img_file:
                    img = PIL.Image.open(os.path.join(output_dir, file_name))
                    
                    # Gọi API gemini
                    response = model.generate_content([prompt, img], stream=True)
                    response.resolve()

                    # Lấy mô tả từ response
                    description = response.text
                    
                    # Ghi kết quả vào file output
                    output.write(f'Description for {file_name}:\n')
                    output.write(description)
                    output.write('\n\n')  # Thêm dòng trống giữa các mô tả

                    time.sleep(4)
            else:
                print(f'File {file_name} does not exist.')

    print("done!!")
                
# def api_call_json(output_dir):

        
def main():
    load_dotenv()
    genai.configure(api_key=os.getenv('gemini_api_key'))

    video_path = 'aslvid.mp4'
    output_dir = 'frames'
    model_path = 'aslmodel.pth'

    frame_cut(output_dir, video_path)
    # api_call_txt(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # frame_cut(output_dir, video_path)

    model = load_model(model_path, device)
    result_string = process_frames(model, 'frames', device)
    # for char in result_string:
        
    print("Resulting String:", result_string)

                
if __name__ == "__main__":
    main()