def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."
caption = preprocess_caption(caption=caption)


attention_mask = (torch.eye(num_token, device=input_ids.device).bool().unsqueeze(0).repeat(bs, 1, 1))
position_ids = torch.zeros((bs, num_token), device=input_ids.device)
cate_to_token_mask_list = [[] for _ in range(bs)]
previous_col = 0
for i in range(idxs.shape[0]):
  row, col = idxs[i]
  if (col == 0) or (col == num_token - 1):
     attention_mask[row, col, col] = True
     position_ids[row, col] = 0
  else:
     attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
     position_ids[row, previous_col + 1 : col + 1] = torch.arange(0, col - previous_col, device=input_ids.device)
     c2t_maski = torch.zeros((num_token), device=input_ids.device).bool()
     c2t_maski[previous_col + 1 : col] = True
     cate_to_token_mask_list[row].append(c2t_maski)
     previous_col = col
cate_to_token_mask_list = [torch.stack(cate_to_token_mask_listi, dim=0) for cate_to_token_mask_listi in cate_to_token_mask_list]
return attention_mask, position_ids.to(torch.long), cate_to_token_mask_list

# tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(samples.device)

"""
bs, num_token = input_ids.shape
special_tokens_mask = torch.zeros((bs, num_token), device=input_ids.device).bool()
for special_token in special_tokens_list:
   special_tokens_mask |= input_ids == special_token
"""

"""
def predict(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption = preprocess_caption(caption=caption)
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    

    
"""