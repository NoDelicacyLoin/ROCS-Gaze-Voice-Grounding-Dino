import os
import torch
from PIL import Image
from torchvision import transforms

from groundingdino.util.inference import load_model
from groundingdino.util.utils import get_phrases_from_posmap
from models.GroundingDINO.groundingdino import GroundingDINO

from groundingdino.util.slconfig import SLConfig
from groundingdino.util.nested_tensor import nested_tensor_from_tensor_list


# === 加载图像与热力图 ===
def load_image_and_heatmap(image_path, heatmap_path):
    image = Image.open(image_path).convert("RGB")
    heatmap = Image.open(heatmap_path).convert("L")  # 单通道灰度
    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
    ])
    return transform(image), transform(heatmap)


# === 文本字典 ===
def make_text_dict(caption: str):
    return {"caption": caption}


# === 加载模型 ===
def load_custom_model(cfg_path, ckpt_path):
    args = SLConfig.fromfile(cfg_path)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GroundingDINO(args)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    model = model.to(args.device)
    return model


# === 推理主逻辑 ===
@torch.no_grad()
def run_inference(model, image_tensor, heatmap_tensor, caption):
    device = next(model.parameters()).device
    samples = nested_tensor_from_tensor_list([image_tensor]).to(device)
    gaze_features = [heatmap_tensor.to(device)]
    text_dict = make_text_dict(caption)

    outputs = model(samples, text_dict=text_dict, gaze_features=gaze_features)

    if "pred_boxes" in outputs:
        print("预测框数量：", outputs["pred_boxes"].shape[1])
    else:
        print("模型输出：", outputs.keys())

    return outputs


# === 入口 ===
if __name__ == "__main__":
    # 示例路径，替换成你自己的
    image_path = "demo/test.jpg"
    heatmap_path = "demo/test_heatmap.png"
    caption = "The UNITED STATES AIR FORCE text"

    config_path = "config/cfg_odvg.py"
    checkpoint_path = "output/checkpoint.pth"

    image_tensor, heatmap_tensor = load_image_and_heatmap(image_path, heatmap_path)
    model = load_custom_model(config_path, checkpoint_path)

    result = run_inference(model, image_tensor, heatmap_tensor, caption)
