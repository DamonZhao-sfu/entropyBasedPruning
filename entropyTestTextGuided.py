import torch
import torch.nn.functional as F
from PIL import Image
from cdencoder import CLIPVisionTower
from transformers import LlavaForConditionalGeneration, LlavaProcessor, CLIPVisionModel, CLIPImageProcessor
import time
import torch
import numpy as np
from PIL import Image
import requests
import json
import io
import base64
import os
import csv
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from datetime import datetime
from transformers import LlavaForConditionalGeneration, LlavaProcessor, CLIPVisionModel, CLIPImageProcessor
from cdencoder import CLIPVisionTower

vision_tower_name = "/data/models/clip-vit-p14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
class MockArgs:
    def __init__(self):
        self.mm_vision_select_layer = -2
        self.mm_vision_select_feature = 'patch'

mock_args = MockArgs()
vision_tower = CLIPVisionTower(vision_tower_name, mock_args, delay_load=False)
vision_tower = vision_tower.to("cuda")


def encode_image_embedding_to_base64(image_embedding):
    """
    Encode image embedding tensor to base64 string
    
    Args:
        image_embedding: PyTorch tensor containing image embeddings
        
    Returns:
        base64 encoded string of the tensor
    """
    buffer = io.BytesIO()
    torch.save(image_embedding, buffer)
    buffer.seek(0)
    binary_data = buffer.read()
    base64_image_embedding = base64.b64encode(binary_data).decode('utf-8')
    return base64_image_embedding


def call_vllm_api_with_embeds(image_embedding, question="What's in this image?", model="llava-hf/llava-1.5-7b-hf", api_url="http://localhost:8005"):
    """
    Call vLLM API with encoded image embedding
    
    Args:
        image_embedding: PyTorch tensor containing image embeddings
        question: Question to ask about the image
        model: Model name to use
        api_url: API endpoint URL
        
    Returns:
        API response
    """
    # Encode image embedding
    base64_image_embedding = encode_image_embedding_to_base64(image_embedding)
    
    # Prepare the request payload
    embeds = {
        "type": "image_embeds",
        "image_embeds": base64_image_embedding
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    },
                    embeds,
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "guided_choice": ["A", "B", "C", "D"]  # Add this line
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Sending request to {api_url}/v1/chat/completions...")
        response = requests.post(
            f"{api_url}/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling vLLM API: {e}")
        return None



def call_vllm_api_with_image_base64(image_path, question="What's in this image?", model="llava-hf/llava-1.5-7b-hf", api_url="http://localhost:8005"):
    """
    Call vLLM API with direct image path using base64 encoding
    
    Args:
        image_path: Path to the image file (local path or URL)
        question: Question to ask about the image
        model: Model name to use
        api_url: API endpoint URL
        
    Returns:
        API response
    """
    try:
        # Read and encode image to base64
        if image_path.startswith(('http://', 'https://')):
            # Download image from URL
            response = requests.get(image_path)
            response.raise_for_status()
            image_data = response.content
        else:
            # Read local image file
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
        
        # Convert to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Detect image format from file extension or content
        if image_path.lower().endswith(('.png', '.PNG')):
            image_format = 'png'
        elif image_path.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')):
            image_format = 'jpeg'
        elif image_path.lower().endswith(('.gif', '.GIF')):
            image_format = 'gif'
        elif image_path.lower().endswith(('.webp', '.WEBP')):
            image_format = 'webp'
        else:
            # Default to jpeg if format cannot be determined
            image_format = 'jpeg'
        
        # Create data URL with base64 encoded image
        image_url = f"data:image/{image_format};base64,{base64_image}"
        
    except Exception as e:
        print(f"Error reading/encoding image: {e}")
        return None
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "guided_choice": ["A", "B", "C", "D"]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Sending request to {api_url}/v1/chat/completions with base64 encoded image")
        response = requests.post(
            f"{api_url}/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling vLLM API: {e}")
        return None

def call_vllm_api_with_image_path(image_path, question="What's in this image?", model="llava-hf/llava-1.5-7b-hf", api_url="http://localhost:8005"):
    """
    Call vLLM API with direct image path/URL
    
    Args:
        image_path: Path to the image file (local path or URL)
        question: Question to ask about the image
        model: Model name to use
        api_url: API endpoint URL
        
    Returns:
        API response
    """
    # Convert local path to file URL if it's a local path
    if os.path.exists(image_path) and not image_path.startswith(('http://', 'https://')):
        # Convert local path to file URL
        image_url = f"file://{os.path.abspath(image_path)}"
    else:
        # Assume it's already a URL
        image_url = image_path
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "guided_choice": ["A", "B", "C", "D"]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Sending request to {api_url}/v1/chat/completions with image URL: {image_url}")
        response = requests.post(
            f"{api_url}/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling vLLM API: {e}")
        return None

# Wrapper function to choose between embed and image path methods
def call_vllm_api(image_embedding=None, image_path=None, question="What's in this image?", model="llava-hf/llava-1.5-7b-hf", api_url="http://localhost:8005", use_image_path=False):
    """
    Call vLLM API with either image embedding or direct image path
    
    Args:
        image_embedding: PyTorch tensor containing image embeddings (for embed method)
        image_path: Path to image file (for direct path method)
        question: Question to ask about the image
        model: Model name to use
        api_url: API endpoint URL
        use_image_path: If True, use direct image path method; if False, use embedding method
        
    Returns:
        API response
    """
    if use_image_path:
        if image_path is None:
            raise ValueError("image_path must be provided when use_image_path=True")
        return call_vllm_api_with_image_path(image_path, question, model, api_url)
    else:
        if image_embedding is None:
            raise ValueError("image_embedding must be provided when use_image_path=False")
        return call_vllm_api_with_embeds(image_embedding, question, model, api_url)



def adjust_pruning_rate(scores, base_keep_ratio=0.125):
    # Ensure scores are valid and not all the same
    if torch.isnan(scores).any() or torch.isinf(scores).any():
        print("Warning: Invalid scores detected, using base keep ratio")
        return base_keep_ratio
    
    # Handle case where all scores are identical
    if torch.allclose(scores, scores[0]):
        print("Warning: All scores are identical, using base keep ratio")
        return base_keep_ratio
    
    # Compute softmax probabilities with numerical stability
    # Subtract max for numerical stability
    scores_stable = scores - scores.max()
    probs = F.softmax(scores_stable, dim=0)
    
    # Add small epsilon to prevent log(0)
    eps = 1e-10
    probs = torch.clamp(probs, min=eps, max=1.0)
    
    # Compute entropy: H = -sum(p * log(p))
    log_probs = torch.log(probs)
    entropy = -torch.sum(probs * log_probs)
    
    # Normalize entropy by maximum possible entropy (log(n))
    max_entropy = torch.log(torch.tensor(len(scores), dtype=scores.dtype, device=scores.device))
    normalized_entropy = entropy / max_entropy
    
    print(f"Raw entropy: {entropy.item():.4f}")
    print(f"Max possible entropy: {max_entropy.item():.4f}")
    print(f"Normalized entropy: {normalized_entropy.item():.4f}")
    
    # Adjust keep ratio based on normalized entropy
    # High entropy (uniform distribution) -> keep more tokens
    # Low entropy (peaked distribution) -> can prune more aggressively
    if normalized_entropy < 0.3:  # Very peaked distribution
        adjusted_ratio = base_keep_ratio * 0.7  # More aggressive pruning
        print("Low entropy detected - more aggressive pruning")
    elif normalized_entropy > 0.8:  # More uniform distribution
        adjusted_ratio = base_keep_ratio * 1.3  # Less aggressive pruning
        print("High entropy detected - less aggressive pruning")
    else:  # Medium entropy
        adjusted_ratio = base_keep_ratio
        print("Medium entropy - using base pruning rate")
    
    # Ensure the ratio stays within reasonable bounds
    adjusted_ratio = torch.clamp(torch.tensor(adjusted_ratio), min=0.05, max=0.5).item()
    
    print(f"Base keep ratio: {base_keep_ratio:.3f}")
    print(f"Adjusted keep ratio: {adjusted_ratio:.3f}")
    
    return adjusted_ratio



def compute_pruning_scores(attention_matrix, lambda_val, alpha=0.5):
    eps = 1e-10
        
    attention_sum = attention_matrix.sum(dim=0, keepdim=True)
    attention_per_image_token = attention_matrix / (attention_sum + eps)
    attention_per_image_token = torch.clamp(attention_per_image_token, min=eps)
    
    log_attention = torch.log(attention_per_image_token)
    H = -(attention_per_image_token * log_attention).sum(dim=0)
    
    scores = - lambda_val * H
    return scores


# def getPrunedVisualToken(model, image_path, texts, keep_ratio=0.125, lambda_val=0.1, recovery_ratio=0.1):
#     # Load and preprocess image
#     image = Image.open(image_path)
#     inputs = vision_tower.image_processor(image, return_tensors="pt")
#     images = inputs["pixel_values"]
#     image_stream = torch.cuda.Stream()
#     text_stream = torch.cuda.Stream()
    
#     model_device = vision_tower.device
    
#     # Process image features
#     with torch.cuda.stream(image_stream):
#         image_forward_outs = vision_tower.vision_tower(
#             images.to(device=model_device, dtype=vision_tower.dtype),
#             output_hidden_states=True,
#             output_attentions=True
#         )
#         image_outputs = vision_tower.feature_select(image_forward_outs)
#         image_features = image_outputs.to(images.dtype)
    
#     # Process text embeddings
#     if texts is not None:
#         with torch.cuda.stream(text_stream):
#             text_inputs = vision_tower.text_tokenizer(
#                 text=texts, return_tensors="pt"
#             )
#             text_segment = (text_inputs.input_ids.shape[1] - 1) // vision_tower.max_position_embeddings + 1
#             text_padding = vision_tower.max_position_embeddings * text_segment - text_inputs.input_ids.shape[1]
#             text_inputs = {
#                 k: torch.cat([v, v.new_zeros((v.shape[0], text_padding))], 
#                             dim=1).reshape(-1, vision_tower.max_position_embeddings).to(device=model_device)
#                 for k, v in text_inputs.items()
#             }
#             text_embeds = vision_tower.text_tower(**text_inputs).text_embeds
    
#             if text_embeds.dim() == 2:
#                 text_embeds = text_embeds.unsqueeze(0)  # Add batch dimension if missing
#             elif text_embeds.dim() > 3:
#                 # Handle cases where output might have extra dimensions
#                 text_embeds = text_embeds.squeeze().reshape(1, -1, text_embeds.size(-1))
    
#     torch.cuda.synchronize()
    
#     B, N, C = image_features.shape
#     image_features = image_features.to(device=model_device, dtype=torch.float16)
#     model.multi_modal_projector = model.multi_modal_projector.to(model_device)
#     image_features = model.multi_modal_projector(image_features)
    
#     projected_dim = image_features.shape[-1]
#     if texts is not None:
#         # Ensure text_embeds and image_features are aligned in dimensionality
#         text_embeds = text_embeds.to(device=model_device, dtype=torch.float16)

#         # Check if dimensions match, if not, we need to project text_embeds
#         if text_embeds.shape[-1] != image_features.shape[-1]:
#             # Create a simple linear projection
#             projection_layer = torch.nn.Linear(text_embeds.shape[-1], image_features.shape[-1]).to(model_device).half()
#             text_embeds_projected = projection_layer(text_embeds)
                    
#             # Compute similarity using projected text embeddings
#             attention_matrix = torch.bmm(text_embeds_projected, image_features.transpose(1, 2))  # [B, num_text_tokens, num_image_tokens]
#         else:
#             # Dimensions already match
#             attention_matrix = torch.bmm(text_embeds, image_features.transpose(1, 2))  # [B, num_text_tokens, num_image_tokens]
        
#         attention_matrix = F.softmax(attention_matrix, dim=-1)  
#         attention_matrix = attention_matrix.mean(dim=0)  
        
#     scores = compute_pruning_scores(attention_matrix, lambda_val)
    
#     # Select top-k tokens
#     num_image_tokens = N
#     num_tokens_to_keep = min(int(keep_ratio * num_image_tokens), num_image_tokens)
#     _, top_indices = torch.topk(scores, num_tokens_to_keep, dim=-1)
    
#     # Validate indices
#     if (top_indices >= num_image_tokens).any() or (top_indices < 0).any():
#         raise ValueError(f"top_indices contains invalid values: {top_indices}")
    
#     all_indices = torch.arange(num_image_tokens, device=model_device)
#     pruned_indices = all_indices[~torch.isin(all_indices, top_indices)]
    
#     if pruned_indices.numel() > 0:
#         summary_token = image_features[:, pruned_indices, :].mean(dim=1, keepdim=True)  # [B, 1, C]
#         image_features_selected = torch.cat(
#             [image_features[:, top_indices, :], summary_token], dim=1
#         )  # [B, num_tokens_to_keep + 1, C]
#     else:
#         image_features_selected = image_features[:, top_indices, :]  # [B, num_tokens_to_keep, C]
    
#     # Recover additional tokens based on text relevance
#     num_tokens_to_recover = min(int(recovery_ratio * num_image_tokens), pruned_indices.numel())
#     if num_tokens_to_recover > 0:
#         recovery_scores = attention_matrix.sum(dim=0)[pruned_indices]  # [num_pruned_tokens]
#         _, recovery_indices = torch.topk(recovery_scores, num_tokens_to_recover, dim=-1)
#         recovery_indices = pruned_indices[recovery_indices]  # Map back to original indices
#         image_features_recovered = image_features[:, recovery_indices, :]  # [B, num_tokens_to_recover, C]
#         image_features_selected = torch.cat(
#             [image_features_selected, image_features_recovered], dim=1
#         )  # [B, num_tokens_to_keep + 1 + num_tokens_to_recover, C]
    
#     image_features_selected = image_features_selected.detach().cpu()
#     print(f"Final output shape: {image_features_selected.shape}")
    
#     return image_features_selected



def getOriginalVisualToken(model, image_path, texts, keep_ratio=0.25, lambda_val=0.1, recovery_ratio=0.1):
    # Load and preprocess image
    image = Image.open(image_path)
    inputs = vision_tower.image_processor(image, return_tensors="pt")
    images = inputs["pixel_values"]
    image_stream = torch.cuda.Stream()
    text_stream = torch.cuda.Stream()
    
    model_device = vision_tower.device
    
    # Process image features
    with torch.cuda.stream(image_stream):
        image_forward_outs = vision_tower.vision_tower(
            images.to(device=model_device, dtype=vision_tower.dtype),
            output_hidden_states=True,
            output_attentions=True
        )
        image_outputs = vision_tower.feature_select(image_forward_outs)
        image_features = image_outputs.to(images.dtype)
      
    torch.cuda.synchronize()
    
    B, N, C = image_features.shape
    image_features = image_features.to(device=model_device, dtype=torch.float16)
    model.multi_modal_projector = model.multi_modal_projector.to(model_device)
    image_features = model.multi_modal_projector(image_features).detach().cpu()
    return image_features

def save_results_to_csv(results_data, filename="tpcds_query_times.csv"):
    """
    Save query results data to CSV file with correct headers
    
    Args:
        results_data: List of dictionaries containing query execution data
        filename: Output CSV filename
    """
    if not results_data:
        print("No results data to save")
        return
    
    # Define CSV headers that match the query data structure from parse_query_file()
    headers = [
        'test_number',
        'total_tests', 
        'sample_image_path',
        'timestamp',
        'embed_time',
        'api_call_time',
        'total_time',
        'api_success',
        'generated_text',
        'predicted_answer',
        'is_correct',
        'error_message',
        'full_response',
        'model_path',
        'original_token',
        'pruned_token',
        'api_url',
        'method',
        'preprocess_time',
        'encode_time',
        'project_time',
        'prune_time'
    ]
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            
            for row in results_data:
                # Fill missing fields with empty strings
                complete_row = {header: row.get(header, '') for header in headers}
                writer.writerow(complete_row)
        
        print(f"Results saved to {filename}")
        print(f"Number of records saved: {len(results_data)}")
        
        # Verify file was created and has content
        import os
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"File size: {file_size} bytes")
        
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        import traceback
        traceback.print_exc()

def getPrunedVisualTokenVisPruner_textguided_optimized(model, image_path, texts, keep_ratio=0.125, 
                                                      important_ratio=0.6, recovery_ratio=0.1,
                                                      text_guidance_weight=0.3, approximation_k=128):
    # Convert binary data to PIL Image
    begin = time.time()
    image = Image.open(image_path)
    inputs = vision_tower.image_processor(image, return_tensors="pt")
    images = inputs["pixel_values"]
    
    model_device = vision_tower.device
    dtype = vision_tower.dtype
    
    # Process image features and get attention from visual encoder
    with torch.no_grad():
        image_forward_outs = vision_tower.vision_tower(
            images.to(device=model_device, dtype=dtype),
            output_hidden_states=True,
            output_attentions=True
        )
        
        # Extract [CLS] attention more efficiently
        attentions = image_forward_outs.attentions
        if len(attentions) > 1:
            cls_attention = attentions[-2].squeeze(0).mean(dim=0)[0, 1:]  # [num_patches]
        else:
            cls_attention = attentions[-1].squeeze(0).mean(dim=0)[0, 1:]
        
        image_outputs = vision_tower.feature_select(image_forward_outs)
        image_features = image_outputs.to(dtype)  # Keep on GPU
    
    B, N, C = image_features.shape
    
    # Ensure consistent dtypes
    image_features = image_features.to(device=model_device, dtype=torch.float16)
    model.multi_modal_projector = model.multi_modal_projector.to(device=model_device, dtype=torch.float16)
    
    # Pre-calculate token counts
    num_tokens_to_keep = min(int(keep_ratio * N), N)
    num_important_tokens = int(num_tokens_to_keep * important_ratio)
    num_diverse_tokens = num_tokens_to_keep - num_important_tokens
    
    # Early exit if keeping all tokens
    if num_tokens_to_keep >= N:
        image_features = model.multi_modal_projector(image_features)
        return image_features.detach().cpu()
    
    end = time.time()
    print("preprocess" + str(end-begin))

    # ===== TEXT-GUIDED IMPORTANCE SCORING (Modified Step 1) =====
    begin = time.time()
    if texts is not None and text_guidance_weight > 0:
        # Process text embeddings efficiently
        text_embeds = process_text_efficiently(texts, vision_tower, model_device)
        
        if text_embeds is not None:
            # Apply multimodal projection to get compatible feature space
            projected_features = model.multi_modal_projector(image_features)  # [B, N, hidden_dim]
            
            # Ensure dimension compatibility
            if text_embeds.shape[-1] != projected_features.shape[-1]:
                projection_layer = torch.nn.Linear(
                    text_embeds.shape[-1], projected_features.shape[-1], bias=False
                ).to(model_device, dtype=torch.float16)
                text_embeds = projection_layer(text_embeds)
            
            # Compute text-visual similarity scores
            # Normalize for better similarity computation
            text_embeds_norm = F.normalize(text_embeds, p=2, dim=-1)  # [B, text_tokens, hidden_dim]
            visual_embeds_norm = F.normalize(projected_features, p=2, dim=-1)  # [B, N, hidden_dim]
            
            # Efficient similarity computation: [B, text_tokens, N]
            similarity_scores = torch.bmm(text_embeds_norm, visual_embeds_norm.transpose(-2, -1))
            
            # Aggregate across text tokens (max pooling for strongest alignment)
            text_visual_scores, _ = similarity_scores.max(dim=1)  # [B, N]
            text_visual_scores = text_visual_scores.squeeze(0)  # [N]
            
            # Normalize CLS attention and text-visual scores to [0, 1]
            cls_attention_norm = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
            text_visual_norm = (text_visual_scores - text_visual_scores.min()) / (text_visual_scores.max() - text_visual_scores.min() + 1e-8)
            
            # Hybrid importance scoring
            hybrid_importance = (1 - text_guidance_weight) * cls_attention_norm + text_guidance_weight * text_visual_norm
        else:
            hybrid_importance = cls_attention
            projected_features = model.multi_modal_projector(image_features)
    else:
        hybrid_importance = cls_attention
        projected_features = model.multi_modal_projector(image_features)
    
    # Step 1: Select important tokens using hybrid scoring
    _, important_indices = torch.topk(hybrid_importance, num_important_tokens, dim=-1)
    
    # Create boolean mask for remaining indices
    all_mask = torch.ones(N, dtype=torch.bool, device=model_device)
    all_mask[important_indices] = False
    remaining_indices = torch.nonzero(all_mask, as_tuple=True)[0]
    
    end = time.time()
    print("step 1" + str(end-begin))

    begin = time.time()
    # ===== TEXT-GUIDED DIVERSE SELECTION (Modified Step 2) =====
    diverse_indices = torch.empty(0, dtype=torch.long, device=model_device)
    
    if num_diverse_tokens > 0 and len(remaining_indices) > 0:
        if len(remaining_indices) <= num_diverse_tokens:
            diverse_indices = remaining_indices
        else:
            # For diversity selection, also consider text guidance
            if texts is not None and text_guidance_weight > 0:
                # Weight the remaining features by their text relevance
                remaining_text_scores = text_visual_norm[remaining_indices] if 'text_visual_norm' in locals() else torch.ones(len(remaining_indices), device=model_device)
                
                # Apply weighted sampling probability
                sampling_weights = 0.7 + 0.3 * remaining_text_scores  # Ensure minimum probability
                sampling_probs = sampling_weights / sampling_weights.sum()
                
                # Use approximate sampling for large sets
                if len(remaining_indices) > 500:
                    # Sample based on text relevance + randomness
                    sample_size = min(num_diverse_tokens * 3, len(remaining_indices))
                    sampled_idx = torch.multinomial(sampling_probs, sample_size, replacement=False)
                    sampled_indices = remaining_indices[sampled_idx]
                    
                    remaining_features = projected_features[0, sampled_indices, :]
                    remaining_features = F.normalize(remaining_features, p=2, dim=-1)
                    
                    diverse_idx = similarity_based_duplicate_removal_fast(
                        remaining_features, min(num_diverse_tokens, len(sampled_indices))
                    )
                    diverse_indices = sampled_indices[diverse_idx]
                else:
                    # Full computation with text guidance
                    remaining_features = projected_features[0, remaining_indices, :]
                    remaining_features = F.normalize(remaining_features, p=2, dim=-1)
                    
                    diverse_idx = text_guided_diversity_selection(
                        remaining_features, sampling_weights, num_diverse_tokens
                    )
                    diverse_indices = remaining_indices[diverse_idx]
            else:
                # Original diversity selection without text guidance
                if len(remaining_indices) > 500:
                    sample_size = min(num_diverse_tokens * 4, len(remaining_indices))
                    sampled_idx = torch.randperm(len(remaining_indices), device=model_device)[:sample_size]
                    sampled_indices = remaining_indices[sampled_idx]
                    
                    remaining_features = projected_features[0, sampled_indices, :]
                    remaining_features = F.normalize(remaining_features, p=2, dim=-1)
                    
                    diverse_idx = similarity_based_duplicate_removal_fast(
                        remaining_features, min(num_diverse_tokens, len(sampled_indices))
                    )
                    diverse_indices = sampled_indices[diverse_idx]
                else:
                    remaining_features = projected_features[0, remaining_indices, :]
                    remaining_features = F.normalize(remaining_features, p=2, dim=-1)
                    
                    diverse_idx = similarity_based_duplicate_removal_fast(
                        remaining_features, num_diverse_tokens
                    )
                    diverse_indices = remaining_indices[diverse_idx]
        
    # Combine and sort indices
    selected_indices = torch.cat([important_indices, diverse_indices])
    selected_indices = torch.sort(selected_indices)[0]
    
    # Extract selected features
    image_features_selected = projected_features[:, selected_indices, :]
    
    end = time.time()
    print("step 2" + str(end-begin))

    # ===== APPROXIMATED TEXT-GUIDED RECOVERY (Optimized Step 3) =====
    if texts is not None and recovery_ratio > 0:
        text_embeds = process_text_efficiently(texts, vision_tower, model_device) if 'text_embeds' not in locals() else text_embeds
        
        if text_embeds is not None:
            # Get pruned indices
            selected_mask = torch.zeros(N, dtype=torch.bool, device=model_device)
            selected_mask[selected_indices] = True
            pruned_indices = torch.nonzero(~selected_mask, as_tuple=True)[0]
            
            if len(pruned_indices) > 0:
                num_tokens_to_recover = min(int(recovery_ratio * N), len(pruned_indices))
                
                if num_tokens_to_recover > 0:
                    # ===== APPROXIMATION TECHNIQUES FOR STEP 3 =====
                    
                    # Technique 1: K-means clustering approximation for large token sets
                    if len(pruned_indices) > approximation_k * 2:
                        pruned_features = projected_features[:, pruned_indices, :].squeeze(0)  # [num_pruned, hidden_dim]
                        
                        # Use k-means to find representative clusters
                        centroids, cluster_assignments = kmeans_approximation(
                            pruned_features, min(approximation_k, len(pruned_indices))
                        )
                        
                        # Compute attention only on centroids (much faster)
                        if text_embeds.shape[-1] != centroids.shape[-1]:
                            projection_layer = torch.nn.Linear(
                                text_embeds.shape[-1], centroids.shape[-1], bias=False
                            ).to(model_device, dtype=torch.float16)
                            text_embeds_proj = projection_layer(text_embeds)
                        else:
                            text_embeds_proj = text_embeds
                        
                        # Efficient attention computation on centroids
                        attention_scores_centroids = torch.einsum('btc,kc->btk', text_embeds_proj, centroids)
                        attention_scores_centroids = F.softmax(attention_scores_centroids, dim=-1).mean(dim=(0, 1))  # [k]
                        
                        # Map back to original tokens using cluster assignments
                        attention_scores = attention_scores_centroids[cluster_assignments]  # [num_pruned]
                        
                    else:
                        # Technique 2: Direct computation with optimizations for smaller sets
                        pruned_features = projected_features[:, pruned_indices, :]
                        
                        if text_embeds.shape[-1] != pruned_features.shape[-1]:
                            # Use cached projection if available
                            if not hasattr(model, '_text_projection_cache'):
                                model._text_projection_cache = torch.nn.Linear(
                                    text_embeds.shape[-1], pruned_features.shape[-1], bias=False
                                ).to(model_device, dtype=torch.float16)
                            text_embeds_proj = model._text_projection_cache(text_embeds)
                        else:
                            text_embeds_proj = text_embeds
                        
                        # Use approximate attention (top-k approximation)
                        attention_scores = approximate_attention_topk(
                            text_embeds_proj, pruned_features, top_k=min(64, text_embeds_proj.shape[1])
                        )
                    
                    # Technique 3: Smart recovery using attention + diversity
                    # Recover top tokens but ensure some diversity
                    if num_tokens_to_recover > 1:
                        # Get top candidates (2x more than needed)
                        top_candidates = min(num_tokens_to_recover * 2, len(attention_scores))
                        _, top_candidate_idx = torch.topk(attention_scores, top_candidates)
                        
                        # Among top candidates, select diverse ones
                        candidate_features = projected_features[0, pruned_indices[top_candidate_idx], :]
                        diverse_recovery_idx = similarity_based_duplicate_removal_fast(
                            F.normalize(candidate_features, p=2, dim=-1), num_tokens_to_recover
                        )
                        recovery_local_idx = top_candidate_idx[diverse_recovery_idx]
                    else:
                        _, recovery_local_idx = torch.topk(attention_scores, num_tokens_to_recover)
                    
                    recovery_indices = pruned_indices[recovery_local_idx]
                    
                    # Concatenate recovered features
                    image_features_recovered = projected_features[:, recovery_indices, :]
                    image_features_selected = torch.cat(
                        [image_features_selected, image_features_recovered], dim=1
                    )
    
    # Single GPU-CPU transfer at the end
    result = image_features_selected.detach().cpu()
    
    print(f"Final output shape: {result.shape}")
    print(f"Kept {result.shape[1]} out of {N} tokens ({result.shape[1]/N*100:.1f}%)")
    print(f"Text guidance weight: {text_guidance_weight}")
    
    return result


def text_guided_diversity_selection(features, text_weights, num_tokens):
    """
    Select diverse tokens with text guidance weighting
    """
    selected_indices = []
    available_mask = torch.ones(len(features), dtype=torch.bool, device=features.device)
    
    # Start with highest text-weighted token
    weighted_scores = text_weights.clone()
    
    for _ in range(num_tokens):
        if available_mask.sum() == 0:
            break
            
        # Select token with highest current score
        available_scores = weighted_scores.clone()
        available_scores[~available_mask] = -float('inf')
        selected_idx = available_scores.argmax()
        
        selected_indices.append(selected_idx)
        available_mask[selected_idx] = False
        
        if available_mask.sum() == 0:
            break
        
        # Update scores based on similarity to selected token
        selected_feature = features[selected_idx:selected_idx+1]
        similarities = torch.mm(features, selected_feature.t()).squeeze(-1)
        
        # Reduce scores for similar tokens (diversity penalty)
        penalty = 0.8 * similarities * available_mask.float()
        weighted_scores = weighted_scores - penalty
    
    return torch.tensor(selected_indices, device=features.device)


def kmeans_approximation(features, k, max_iters=10):
    """
    Fast k-means clustering approximation
    """
    n, d = features.shape
    device = features.device
    
    # Initialize centroids randomly
    init_indices = torch.randperm(n, device=device)[:k]
    centroids = features[init_indices].clone()
    
    for _ in range(max_iters):
        # Compute distances efficiently
        distances = torch.cdist(features, centroids)  # [n, k]
        assignments = distances.argmin(dim=1)  # [n]
        
        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for i in range(k):
            mask = assignments == i
            if mask.sum() > 0:
                new_centroids[i] = features[mask].mean(dim=0)
            else:
                new_centroids[i] = centroids[i]  # Keep old centroid if no assignments
        
        # Check convergence (simplified)
        if torch.allclose(centroids, new_centroids, atol=1e-4):
            break
        
        centroids = new_centroids
    
    return centroids, assignments


def approximate_attention_topk(text_embeds, visual_features, top_k=64):
    """
    Compute approximate attention using top-k text tokens only
    """
    B, T, C = text_embeds.shape
    
    if T <= top_k:
        # Use all text tokens
        attention_scores = torch.einsum('btc,bpc->btp', text_embeds, visual_features)
    else:
        # Use only top-k most important text tokens (by norm)
        text_norms = text_embeds.norm(dim=-1)  # [B, T]
        _, topk_indices = torch.topk(text_norms, top_k, dim=-1)  # [B, top_k]
        
        # Gather top-k text embeddings
        topk_text_embeds = torch.gather(
            text_embeds, 1, 
            topk_indices.unsqueeze(-1).expand(-1, -1, C)
        )  # [B, top_k, C]
        
        attention_scores = torch.einsum('btc,bpc->btp', topk_text_embeds, visual_features)
    
    attention_scores = F.softmax(attention_scores, dim=-1).mean(dim=(0, 1))
    return attention_scores


def similarity_based_duplicate_removal_fast(features, num_tokens):
    """
    Fast similarity-based token selection using greedy approach
    """
    if len(features) <= num_tokens:
        return torch.arange(len(features), device=features.device)
    
    selected_indices = []
    available_mask = torch.ones(len(features), dtype=torch.bool, device=features.device)
    
    # Start with random token
    first_idx = torch.randint(0, len(features), (1,), device=features.device)
    selected_indices.append(first_idx.item())
    available_mask[first_idx] = False
    
    # Greedily select most dissimilar tokens
    for _ in range(num_tokens - 1):
        if available_mask.sum() == 0:
            break
        
        # Compute maximum similarity to already selected tokens
        selected_features = features[selected_indices]  # [num_selected, dim]
        available_features = features[available_mask]   # [num_available, dim]
        
        # Compute similarities efficiently
        similarities = torch.mm(available_features, selected_features.t())  # [num_available, num_selected]
        max_similarities, _ = similarities.max(dim=1)  # [num_available]
        
        # Select token with minimum maximum similarity
        min_sim_idx = max_similarities.argmin()
        
        # Map back to original indices
        available_indices = torch.nonzero(available_mask, as_tuple=True)[0]
        selected_original_idx = available_indices[min_sim_idx]
        
        selected_indices.append(selected_original_idx.item())
        available_mask[selected_original_idx] = False
    
    return torch.tensor(selected_indices, device=features.device)

# def process_text_efficiently(texts, vision_tower, device):
#     """
#     Efficient text processing with caching
#     """
#     if texts is None:
#         return None
    
#     # Handle different text input formats
#     if isinstance(texts, str):
#         texts = [texts]
#     elif not isinstance(texts, list):
#         texts = list(texts)
    
#     try:
#         # Use the model's text processing capabilities
#         # This is a placeholder - replace with actual text encoding logic
#         text_inputs = vision_tower.tokenizer(
#             texts, return_tensors="pt", padding=True, truncation=True, max_length=512
#         )
        
#         with torch.no_grad():
#             text_outputs = vision_tower.text_model(**text_inputs.to(device))
#             text_embeds = text_outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
#         return text_embeds
#     except Exception as e:
#         print(f"Text processing failed: {e}")
#         return None


def getPrunedVisualTokenVisPruner_optimized(model, image_path, texts, keep_ratio=0.125, 
                                          important_ratio=0.6, recovery_ratio=0.1, text_guidance_weight=0.5):
    """
    Highly optimized version of VisPruner with multiple speedup techniques:
    1. Minimized GPU-CPU transfers
    2. In-place operations where possible  
    3. Vectorized operations
    4. Memory-efficient tensor operations
    5. Early termination optimizations
    6. Approximate similarity computation for large token sets
    """
    
    # Convert binary data to PIL Image
    image = Image.open(image_path)
    inputs = vision_tower.image_processor(image, return_tensors="pt")
    images = inputs["pixel_values"]
    
    model_device = vision_tower.device
    dtype = vision_tower.dtype
    
    # Process image features and get attention from visual encoder
    with torch.no_grad():
        image_forward_outs = vision_tower.vision_tower(
            images.to(device=model_device, dtype=dtype),
            output_hidden_states=True,
            output_attentions=True
        )
        
        # Extract [CLS] attention more efficiently
        attentions = image_forward_outs.attentions
        if len(attentions) > 1:
            # Use penultimate layer, average across heads in one operation
            cls_attention = attentions[-2].squeeze(0).mean(dim=0)[0, 1:]  # [num_patches]
        else:
            cls_attention = attentions[-1].squeeze(0).mean(dim=0)[0, 1:]
        
        image_outputs = vision_tower.feature_select(image_forward_outs)
        image_features = image_outputs.to(dtype)  # Keep on GPU
    
    B, N, C = image_features.shape
    
    # Ensure consistent dtypes - convert image_features to float16 and projector to same device/dtype
    image_features = image_features.to(device=model_device, dtype=torch.float16)
    model.multi_modal_projector = model.multi_modal_projector.to(device=model_device, dtype=torch.float16)
    projected_features = model.multi_modal_projector(image_features)  # [B, N, hidden_dim]

    # Pre-calculate token counts to avoid repeated calculations
    num_tokens_to_keep = min(int(keep_ratio * N), N)
    num_important_tokens = int(num_tokens_to_keep * important_ratio)
    num_diverse_tokens = num_tokens_to_keep - num_important_tokens
    
    # Early exit if keeping all tokens
    if num_tokens_to_keep >= N:
        image_features = model.multi_modal_projector(image_features)
        return image_features.detach().cpu()
    
    # Step 1: Select important tokens (vectorized topk)
    _, important_indices = torch.topk(cls_attention, num_important_tokens, dim=-1)
    
    # Create boolean mask for remaining indices (more memory efficient)
    all_mask = torch.ones(N, dtype=torch.bool, device=model_device)
    all_mask[important_indices] = False
    remaining_indices = torch.nonzero(all_mask, as_tuple=True)[0]
    
    begin = time.time()
    # Step 2: Optimized diverse token selection
    diverse_indices = torch.empty(0, dtype=torch.long, device=model_device)
    
    if num_diverse_tokens > 0 and len(remaining_indices) > 0:
        if len(remaining_indices) <= num_diverse_tokens:
            diverse_indices = remaining_indices
        else:
            # For diversity selection, also consider text guidance
            if texts is not None and text_guidance_weight > 0:
                # Weight the remaining features by their text relevance
                remaining_text_scores = text_visual_norm[remaining_indices] if 'text_visual_norm' in locals() else torch.ones(len(remaining_indices), device=model_device)
                
                # Apply weighted sampling probability
                sampling_weights = 0.7 + 0.3 * remaining_text_scores  # Ensure minimum probability
                sampling_probs = sampling_weights / sampling_weights.sum()
                
                # Use approximate sampling for large sets
                if len(remaining_indices) > 500:
                    # Sample based on text relevance + randomness
                    sample_size = min(num_diverse_tokens * 3, len(remaining_indices))
                    sampled_idx = torch.multinomial(sampling_probs, sample_size, replacement=False)
                    sampled_indices = remaining_indices[sampled_idx]
                    
                    remaining_features = projected_features[0, sampled_indices, :]
                    remaining_features = F.normalize(remaining_features, p=2, dim=-1)
                    
                    diverse_idx = similarity_based_duplicate_removal_fast(
                        remaining_features, min(num_diverse_tokens, len(sampled_indices))
                    )
                    diverse_indices = sampled_indices[diverse_idx]
                else:
                    # Full computation with text guidance
                    remaining_features = projected_features[0, remaining_indices, :]
                    remaining_features = F.normalize(remaining_features, p=2, dim=-1)
                    
                    diverse_idx = text_guided_diversity_selection(
                        remaining_features, sampling_weights, num_diverse_tokens
                    )
                    diverse_indices = remaining_indices[diverse_idx]
            else:
                # Original diversity selection without text guidance
                if len(remaining_indices) > 500:
                    sample_size = min(num_diverse_tokens * 4, len(remaining_indices))
                    sampled_idx = torch.randperm(len(remaining_indices), device=model_device)[:sample_size]
                    sampled_indices = remaining_indices[sampled_idx]
                    
                    remaining_features = projected_features[0, sampled_indices, :]
                    remaining_features = F.normalize(remaining_features, p=2, dim=-1)
                    
                    diverse_idx = similarity_based_duplicate_removal_fast(
                        remaining_features, min(num_diverse_tokens, len(sampled_indices))
                    )
                    diverse_indices = sampled_indices[diverse_idx]
                else:
                    remaining_features = projected_features[0, remaining_indices, :]
                    remaining_features = F.normalize(remaining_features, p=2, dim=-1)
                    
                    diverse_idx = similarity_based_duplicate_removal_fast(
                        remaining_features, num_diverse_tokens
                    )
                    diverse_indices = remaining_indices[diverse_idx]
        
    # Combine and sort indices
    selected_indices = torch.cat([important_indices, diverse_indices])
    selected_indices = torch.sort(selected_indices)[0]
    
    # Extract selected features
    image_features_selected = projected_features[:, selected_indices, :]
    
    end = time.time()
    print("step 2" + str(end-begin))
    if texts is not None and recovery_ratio > 0:
        # Process text more efficiently
        text_embeds = process_text_efficiently(texts, vision_tower, model_device)
        
        if text_embeds is not None:
            # Get pruned indices using boolean indexing
            selected_mask = torch.zeros(N, dtype=torch.bool, device=model_device)
            selected_mask[selected_indices] = True
            pruned_indices = torch.nonzero(~selected_mask, as_tuple=True)[0]
            
            if len(pruned_indices) > 0:
                num_tokens_to_recover = min(int(recovery_ratio * N), len(pruned_indices))
                
                if num_tokens_to_recover > 0:
                    # Efficient attention computation
                    if text_embeds.shape[-1] != projected_features.shape[-1]:  # FIX: Compare with projected_features
                        # Use smaller projection layer
                        projection_layer = torch.nn.Linear(
                            text_embeds.shape[-1], projected_features.shape[-1],  # FIX: Use projected_features dimension
                            bias=False  # Remove bias for speed
                        ).to(model_device, dtype=torch.float16)
                        text_embeds = projection_layer(text_embeds)
                    
                    # Compute attention only for pruned tokens (memory efficient)
                    pruned_features = projected_features[:, pruned_indices, :]  # FIX: Use projected_features instead of image_features
                    attention_scores = torch.einsum('btc,bpc->btp', text_embeds, pruned_features)
                    attention_scores = F.softmax(attention_scores, dim=-1).mean(dim=(0, 1))
                    
                    # Recover tokens
                    _, recovery_idx = torch.topk(attention_scores, num_tokens_to_recover)
                    recovery_indices = pruned_indices[recovery_idx]
                    
                    # Concatenate efficiently - FIX: Use projected_features for recovery
                    image_features_recovered = projected_features[:, recovery_indices, :]
                    image_features_selected = torch.cat(
                        [image_features_selected, image_features_recovered], dim=1
                    )
    # Single GPU-CPU transfer at the end
    result = image_features_selected.detach().cpu()
    
    print(f"Final output shape: {result.shape}")
    print(f"Kept {result.shape[1]} out of {N} tokens ({result.shape[1]/N*100:.1f}%)")
    
    return result


def process_text_efficiently(texts, vision_tower, model_device):
    """
    Optimized text processing with minimal memory allocation.
    """
    try:
        with torch.no_grad():
            text_inputs = vision_tower.text_tokenizer(text=texts, return_tensors="pt")
            
            # More efficient padding computation
            input_length = text_inputs.input_ids.shape[1]
            max_pos = vision_tower.max_position_embeddings
            
            if input_length <= max_pos:
                # No segmentation needed
                padding_needed = max_pos - input_length
                if padding_needed > 0:
                    pad_tensor = torch.zeros((1, padding_needed), dtype=text_inputs.input_ids.dtype)
                    text_inputs = {
                        k: torch.cat([v, pad_tensor], dim=1).to(model_device)
                        for k, v in text_inputs.items()
                    }
                else:
                    text_inputs = {k: v.to(model_device) for k, v in text_inputs.items()}
            else:
                # Efficient segmentation
                text_segment = (input_length - 1) // max_pos + 1
                total_length = max_pos * text_segment
                padding_needed = total_length - input_length
                
                text_inputs = {
                    k: torch.cat([v, v.new_zeros((v.shape[0], padding_needed))], dim=1)
                    .reshape(-1, max_pos).to(model_device)
                    for k, v in text_inputs.items()
                }
            
            text_embeds = vision_tower.text_tower(**text_inputs).text_embeds
            
            # Efficient reshaping
            if text_embeds.dim() == 2:
                text_embeds = text_embeds.unsqueeze(0)
            elif text_embeds.dim() > 3:
                batch_size = 1
                seq_len = text_embeds.numel() // (batch_size * text_embeds.size(-1))
                text_embeds = text_embeds.view(batch_size, seq_len, -1)
            
            return text_embeds.to(torch.float16)
            
    except Exception as e:
        print(f"Text processing failed: {e}")
        return None


if __name__ == "__main__":
    MODEL_PATH = "/data/models/llava-1.5-7b-hf"
    IMAGE_PATH = "/home/haikai/MMbench/extracted_images/0.jpg"
    TEXTS = "Describe the main object in the image"
    API_URL = "http://localhost:8005"

    PRUNING_CONFIGS = [
        {'keep_ratio': 0.05, 'recovery_ratio': 0.05},
        {'keep_ratio': 0.05, 'recovery_ratio': 0.1},
        {'keep_ratio': 0.05, 'recovery_ratio': 0.2},
        {'keep_ratio': 0.1, 'recovery_ratio': 0.05},
        {'keep_ratio': 0.1, 'recovery_ratio': 0.1},
        {'keep_ratio': 0.1, 'recovery_ratio': 0.2},
        {'keep_ratio': 0.25, 'recovery_ratio': 0.05},
        {'keep_ratio': 0.25, 'recovery_ratio': 0.1},
        {'keep_ratio': 0.25, 'recovery_ratio': 0.2},
    ]

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="cuda",
        attn_implementation="eager"
    )

    processor = LlavaProcessor.from_pretrained(MODEL_PATH, patch_size=14)
    print("Comparing different token pruning methods...")    
    
    # Get sample images for testing
    spark = SparkSession.builder.appName("AudioVisualQAProcessor") \
        .master("local[*]") \
        .config("spark.driver.memory", "16g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    # Load CSV data with questions and hints
    df = spark.read.parquet("/home/haikai/MMbench/dev-00000-of-00001.parquet")
    df.show()

    print("Extracting sample data for testing...")
    sample_data = df.select(
        col("index"),
        col("question"), 
        col("hint"), 
        col("answer"),
        col("A"),
        col("B"), 
        col("C"),
        col("D"),
    ).collect()
    
    print(f"Testing {len(sample_data)} questions with {len(PRUNING_CONFIGS)} configurations...")
    print(f"Total evaluations: {len(sample_data) * len(PRUNING_CONFIGS)}")
    print("-" * 80)

    # Process each configuration separately
    for config_idx, config in enumerate(PRUNING_CONFIGS):
        keep_ratio = config['keep_ratio']
        recovery_ratio = config['recovery_ratio']
        
        print(f"\n{'='*80}")
        print(f"PROCESSING CONFIGURATION {config_idx+1}/{len(PRUNING_CONFIGS)}")
        print(f"keep_ratio={keep_ratio}, recovery_ratio={recovery_ratio}")
        print(f"{'='*80}")
        
        # Initialize results storage for this configuration
        config_results = []
        
        # Run tests for this configuration
        for i, row in enumerate(sample_data, 1):
            question = row['question'] if row['question'] else ""
            hint = row['hint'] if row['hint'] else ""
            correct_answer = row['answer'] if row['answer'] else ""
            option_a = row['A'] if row['A'] else ""
            option_b = row['B'] if row['B'] else ""
            option_c = row['C'] if row['C'] else ""
            option_d = row['D'] if row['D'] else ""
            image_path = "/home/haikai/MMbench/extracted_images/" + str(i-1) + ".jpg" if row['index'] else ""
            
            # Format the complete question with options
            formatted_question = f"Question: {question}\n\nHint: {hint}\n\nOptions:\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\n\nPlease analyze the image and answer the question."

            print(f"Test {i}/{len(sample_data)}: {image_path}")
            print(f"Question: {question}")
            print(f"Correct Answer: {correct_answer}")
            
            # Initialize result record for this iteration
            result_record = {
                'test_number': i,
                'total_tests': len(sample_data),
                'sample_image_path': image_path,
                'keep_ratio': keep_ratio,
                'recovery_ratio': recovery_ratio,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'embed_time': 0,
                'api_call_time': 0,
                'total_time': 0,
                'api_success': False,
                'generated_text': '',
                'predicted_answer': '',
                'is_correct': False,
                'error_message': '',
                'full_response': '',
                'model_path': MODEL_PATH,
                'original_token': 576,
                'pruned_token': 0,
                'api_url': API_URL,
                'method': 'pruned_embeddings',
                'question': question,
                'correct_answer': correct_answer
            }
            
            try:
                prune_time_begin = time.time()
                reduced_tokens = getPrunedVisualTokenVisPruner_optimized(
                    model, 
                    image_path, 
                    formatted_question,
                    keep_ratio=keep_ratio,
                    recovery_ratio=recovery_ratio
                )
                prune_time_end = time.time()

                embed_time = prune_time_end - prune_time_begin
                result_record['original_token'] = 576
                result_record['pruned_token'] = reduced_tokens.shape[1]
                result_record['embed_time'] = embed_time

                # Call API
                api_time_begin = time.time()
                response = call_vllm_api(
                    image_embedding=reduced_tokens.to(torch.float16),
                    question=formatted_question,
                    model="/data/models/llava-1.5-7b-hf",
                    api_url=API_URL,
                    use_image_path=False
                )
                api_time_end = time.time()
                
                api_call_time = api_time_end - api_time_begin
                result_record['api_call_time'] = api_call_time
                result_record['total_time'] = embed_time + api_call_time
                
                if response:
                    result_record['api_success'] = True
                    print(f"embed time: {embed_time:.2f} seconds")
                    print(f"api call time: {api_call_time:.2f} seconds")
                    print(f"pruned tokens: {result_record['pruned_token']}")
                    print("=" * 60)
                    print("API RESPONSE:")
                    print("=" * 60)
                    
                    if 'choices' in response and len(response['choices']) > 0:
                        content = response['choices'][0]['message']['content']
                        result_record['generated_text'] = content
                        
                        # Extract predicted answer (A, B, C, or D)
                        predicted_answer = ""
                        content_upper = content.upper().strip()
                        if content_upper in ['A', 'B', 'C', 'D']:
                            predicted_answer = content_upper
                        elif 'A)' in content_upper or content_upper.startswith('A'):
                            predicted_answer = 'A'
                        elif 'B)' in content_upper or content_upper.startswith('B'):
                            predicted_answer = 'B'
                        elif 'C)' in content_upper or content_upper.startswith('C'):
                            predicted_answer = 'C'
                        elif 'D)' in content_upper or content_upper.startswith('D'):
                            predicted_answer = 'D'
                        
                        result_record['predicted_answer'] = predicted_answer
                        result_record['is_correct'] = (predicted_answer == correct_answer.upper())
                        
                        print(f"Generated text: {content}")
                        print(f"Predicted answer: {predicted_answer}")
                        print(f"Correct answer: {correct_answer}")
                        print(f"Is correct: {result_record['is_correct']}")
                    else:
                        result_record['full_response'] = json.dumps(response, indent=2)
                        print(f"Full response: {json.dumps(response, indent=2)}")
                else:
                    result_record['api_success'] = False
                    result_record['error_message'] = "Failed to get response from vLLM API"
                    print("Failed to get response from vLLM API")
                    
            except Exception as e:
                result_record['api_success'] = False
                result_record['error_message'] = str(e)
                print(f"Error processing test {i}: {e}")
                import traceback
                traceback.print_exc()
            
            # Add the result to our configuration data list
            config_results.append(result_record)
            print("-" * 40)

        # Save results for this configuration immediately
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_filename = f"MMBench_pruning_results_keep{keep_ratio}_recovery{recovery_ratio}_{timestamp}.csv"
        save_results_to_csv(config_results, config_filename)
        
        # Print summary for this configuration
        successful_tests = [r for r in config_results if r['api_success']]
        if successful_tests:
            total_correct = sum(1 for r in successful_tests if r['is_correct'])
            accuracy = total_correct / len(successful_tests)
            avg_embed_time = sum(r['embed_time'] for r in successful_tests) / len(successful_tests)
            avg_api_time = sum(r['api_call_time'] for r in successful_tests) / len(successful_tests)
            avg_pruned_tokens = sum(r['pruned_token'] for r in successful_tests) / len(successful_tests)
            token_reduction = ((576 - avg_pruned_tokens) / 576 * 100)
            
            print(f"\nCONFIGURATION {config_idx+1} SUMMARY:")
            print(f"keep_ratio={keep_ratio}, recovery_ratio={recovery_ratio}")
            print(f"Samples: {len(config_results)}")
            print(f"Successful: {len(successful_tests)}")
            print(f"Accuracy: {accuracy:.2%} ({total_correct}/{len(successful_tests)})")
            print(f"Avg Embed Time: {avg_embed_time:.2f}s")
            print(f"Avg API Time: {avg_api_time:.2f}s")
            print(f"Avg Pruned Tokens: {avg_pruned_tokens:.1f}")
            print(f"Token Reduction: {token_reduction:.1f}%")
            print(f"Results saved to: {config_filename}")
        
        print(f"\n{'-'*80}")
        print(f"Completed configuration {config_idx+1}/{len(PRUNING_CONFIGS)}")
        print(f"{'-'*80}")
    
    print(f"\nAll configurations completed!")
    print(f"Each configuration's results saved to separate CSV files with naming pattern:")
    print(f"pruning_results_keep[RATIO]_recovery[RATIO]_[TIMESTAMP].csv")
