from transformers import AutoImageProcessor, AutoModel
import torch 

def extract_embeddings(model: torch.nn.Module, processor: torch.nn.Module):
    """Utility to compute embeddings."""
    device = model.device

    def pp(batch):
        images = batch["image"]
        
        image_batch_transformed = torch.stack(
            [processor(image, return_tensors='pt')['pixel_values'].squeeze(0) for image in images]
        )
         
        new_batch = {"pixel_values": image_batch_transformed.to(device)}
        with torch.no_grad():
            embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
        return {"embeddings": embeddings}

    return pp


def extract_embedds(model_ckpt, dataset, batch_size=24, device="cuda"):
    
    processor = AutoImageProcessor.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)
    extract_fn = extract_embeddings(model.to(device), processor=processor)
    
    return dataset.map(extract_fn, batched=True, batch_size=batch_size)