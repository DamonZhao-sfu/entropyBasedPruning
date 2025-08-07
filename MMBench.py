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



def adjust_pruning_rate_v4_adaptive_threshold(scores, attention_matrix, base_keep_ratio=0.125):
    """
    Adaptive threshold-based pruning that finds natural breakpoints in score distribution
    """
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    n_tokens = len(sorted_scores)
    
    if n_tokens > 3:
        # Compute first and second derivatives
        first_diff = sorted_scores[:-1] - sorted_scores[1:]
        second_diff = first_diff[:-1] - first_diff[1:]
        
        # Find the point where second derivative is maximum (steepest change)
        if len(second_diff) > 0:
            elbow_idx = torch.argmax(second_diff).item() + 2  # +2 due to diff operations
            elbow_ratio = elbow_idx / n_tokens
        else:
            elbow_ratio = base_keep_ratio
    else:
        elbow_ratio = base_keep_ratio
    
    score_mean = scores.mean()
    score_std = scores.std()
    
    significant_tokens = (scores > (score_mean + 0.5 * score_std)).sum().item()
    statistical_ratio = significant_tokens / n_tokens

    cumulative_importance = torch.cumsum(sorted_scores, dim=0)
    total_importance = cumulative_importance[-1]
    
    # Find how many tokens needed to capture 80% of importance
    importance_threshold = 0.8 * total_importance
    importance_tokens = (cumulative_importance <= importance_threshold).sum().item() + 1
    importance_ratio = importance_tokens / n_tokens
    
    print(f"Elbow ratio: {elbow_ratio:.3f}")
    print(f"Statistical ratio: {statistical_ratio:.3f}")
    print(f"Importance ratio: {importance_ratio:.3f}")
    
    # Combine the three methods
    ratios = [elbow_ratio, statistical_ratio, importance_ratio, base_keep_ratio]
    
    # Use median to avoid extreme values
    adjusted_ratio = torch.median(torch.tensor(ratios)).item()
    
    # Ensure reasonable bounds
    adjusted_ratio = torch.clamp(torch.tensor(adjusted_ratio), min=0.05, max=0.8).item()
        
    return adjusted_ratio

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


def compute_pruning_scores(attention_matrix, lambda_val, alpha=0.5):
    eps = 1e-10
        
    attention_sum = attention_matrix.sum(dim=0, keepdim=True)
    attention_per_image_token = attention_matrix / (attention_sum + eps)
    attention_per_image_token = torch.clamp(attention_per_image_token, min=eps)
    
    log_attention = torch.log(attention_per_image_token)
    H = -(attention_per_image_token * log_attention).sum(dim=0)
    
    scores = - lambda_val * H
    return scores

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


def getPrunedVisualToken(model, image_path, texts, keep_ratio=0.125, lambda_val=0.1, recovery_ratio=0.1):
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
    
    # Process text embeddings
    if texts is not None:
        with torch.cuda.stream(text_stream):
            text_inputs = vision_tower.text_tokenizer(
                text=texts, return_tensors="pt"
            )
            text_segment = (text_inputs.input_ids.shape[1] - 1) // vision_tower.max_position_embeddings + 1
            text_padding = vision_tower.max_position_embeddings * text_segment - text_inputs.input_ids.shape[1]
            text_inputs = {
                k: torch.cat([v, v.new_zeros((v.shape[0], text_padding))], 
                            dim=1).reshape(-1, vision_tower.max_position_embeddings).to(device=model_device)
                for k, v in text_inputs.items()
            }
            text_embeds = vision_tower.text_tower(**text_inputs).text_embeds
    
            if text_embeds.dim() == 2:
                text_embeds = text_embeds.unsqueeze(0)  # Add batch dimension if missing
            elif text_embeds.dim() > 3:
                # Handle cases where output might have extra dimensions
                text_embeds = text_embeds.squeeze().reshape(1, -1, text_embeds.size(-1))
    
    torch.cuda.synchronize()
    
    B, N, C = image_features.shape
    image_features = image_features.to(device=model_device, dtype=torch.float16)
    model.multi_modal_projector = model.multi_modal_projector.to(model_device)
    image_features = model.multi_modal_projector(image_features)
    
    projected_dim = image_features.shape[-1]
    if texts is not None:
        # Ensure text_embeds and image_features are aligned in dimensionality
        text_embeds = text_embeds.to(device=model_device, dtype=torch.float16)

        # Check if dimensions match, if not, we need to project text_embeds
        if text_embeds.shape[-1] != image_features.shape[-1]:
            # Create a simple linear projection
            projection_layer = torch.nn.Linear(text_embeds.shape[-1], image_features.shape[-1]).to(model_device).half()
            text_embeds_projected = projection_layer(text_embeds)
                    
            # Compute similarity using projected text embeddings
            attention_matrix = torch.bmm(text_embeds_projected, image_features.transpose(1, 2))  # [B, num_text_tokens, num_image_tokens]
        else:
            # Dimensions already match
            attention_matrix = torch.bmm(text_embeds, image_features.transpose(1, 2))  # [B, num_text_tokens, num_image_tokens]
        
        attention_matrix = F.softmax(attention_matrix, dim=-1)  
        attention_matrix = attention_matrix.mean(dim=0)  
        
    scores = compute_pruning_scores(attention_matrix, lambda_val)
    
    # Select top-k tokens
    num_image_tokens = N
    num_tokens_to_keep = min(int(keep_ratio * num_image_tokens), num_image_tokens)
    _, top_indices = torch.topk(scores, num_tokens_to_keep, dim=-1)
    
    # Validate indices
    if (top_indices >= num_image_tokens).any() or (top_indices < 0).any():
        raise ValueError(f"top_indices contains invalid values: {top_indices}")
    
    all_indices = torch.arange(num_image_tokens, device=model_device)
    pruned_indices = all_indices[~torch.isin(all_indices, top_indices)]
    
    if pruned_indices.numel() > 0:
        summary_token = image_features[:, pruned_indices, :].mean(dim=1, keepdim=True)  # [B, 1, C]
        image_features_selected = torch.cat(
            [image_features[:, top_indices, :], summary_token], dim=1
        )  # [B, num_tokens_to_keep + 1, C]
    else:
        image_features_selected = image_features[:, top_indices, :]  # [B, num_tokens_to_keep, C]
    
    # Recover additional tokens based on text relevance
    num_tokens_to_recover = min(int(recovery_ratio * num_image_tokens), pruned_indices.numel())
    if num_tokens_to_recover > 0:
        recovery_scores = attention_matrix.sum(dim=0)[pruned_indices]  # [num_pruned_tokens]
        _, recovery_indices = torch.topk(recovery_scores, num_tokens_to_recover, dim=-1)
        recovery_indices = pruned_indices[recovery_indices]  # Map back to original indices
        image_features_recovered = image_features[:, recovery_indices, :]  # [B, num_tokens_to_recover, C]
        image_features_selected = torch.cat(
            [image_features_selected, image_features_recovered], dim=1
        )  # [B, num_tokens_to_keep + 1 + num_tokens_to_recover, C]
    
    image_features_selected = image_features_selected.detach().cpu()
    print(f"Final output shape: {image_features_selected.shape}")
    
    return image_features_selected



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


def generate_timing_summary(results_data):
    """
    Generate and display timing summary statistics
    
    Args:
        results_data: List of dictionaries containing result data
    """
    if not results_data:
        print("No results data to summarize")
        return
    
    # Filter successful API calls
    successful_results = [r for r in results_data if r['api_success']]
    
    if not successful_results:
        print("No successful API calls to summarize")
        return
    
    # Convert to pandas DataFrame for easier analysis
    df = pd.DataFrame(successful_results)
    
    print("\n" + "="*60)
    print("TIMING SUMMARY REPORT")
    print("="*60)
    
    print(f"Total Tests: {len(results_data)}")
    print(f"Successful API Calls: {len(successful_results)}")
    print(f"Success Rate: {len(successful_results)/len(results_data)*100:.1f}%")
    
    # Calculate timing statistics
    timing_stats = {}
    time_columns = ['embed_time', 'api_call_time', 'total_time']
    
    for col in time_columns:
        if col in df.columns:
            timing_stats[col] = {
                'total': df[col].sum(),
                'average': df[col].mean(),
                'min': df[col].min(),
                'max': df[col].max(),
                'std': df[col].std()
            }
    
    print("\n" + "-"*40)
    print("TIMING STATISTICS (seconds)")
    print("-"*40)
    
    for time_type, stats in timing_stats.items():
        print(f"\n{time_type.replace('_', ' ').title()}:")
        print(f"  Total:   {stats['total']:8.2f}s")
        print(f"  Average: {stats['average']:8.2f}s")
        print(f"  Min:     {stats['min']:8.2f}s")
        print(f"  Max:     {stats['max']:8.2f}s")
        print(f"  Std Dev: {stats['std']:8.2f}s")
    
    # Token statistics
    if 'original_tokens' in df.columns and 'reduced_tokens' in df.columns:
        print("\n" + "-"*40)
        print("TOKEN STATISTICS")
        print("-"*40)
        
        orig_tokens = df['original_tokens'].iloc[0] if len(df) > 0 else 0
        avg_reduced = df['reduced_tokens'].mean() if 'reduced_tokens' in df.columns else 0
        reduction_ratio = (orig_tokens - avg_reduced) / orig_tokens * 100 if orig_tokens > 0 else 0
        
        print(f"Original Tokens:     {orig_tokens}")
        print(f"Average Reduced:     {avg_reduced:.1f}")
        print(f"Reduction Ratio:     {reduction_ratio:.1f}%")
    
    # Accuracy statistics
    if 'is_correct' in df.columns:
        correct_predictions = df['is_correct'].sum()
        total_predictions = len([r for r in successful_results if r['predicted_answer']])
        accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
        
        print("\n" + "-"*40)
        print("ACCURACY STATISTICS")
        print("-"*40)
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Total Predictions:   {total_predictions}")
        print(f"Accuracy:           {accuracy:.1f}%")
    
    print("\n" + "="*60)



def getPrunedVisualTokenVisPruner_optimized(model, image_path, texts, keep_ratio=0.125, 
                                          important_ratio=0.6, recovery_ratio=0.1):
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
    
    # Step 2: Optimized diverse token selection
    diverse_indices = torch.empty(0, dtype=torch.long, device=model_device)
    
    if num_diverse_tokens > 0 and len(remaining_indices) > 0:
        if len(remaining_indices) <= num_diverse_tokens:
            # If we need all remaining, skip similarity computation
            diverse_indices = remaining_indices
        else:
            # Apply multimodal projection early to remaining tokens only
            image_features = model.multi_modal_projector.to(model_device)(image_features)
            
            # Use approximate similarity for large token sets (>500 tokens)
            if len(remaining_indices) > 500:
                # Random sampling for approximation - much faster
                sample_size = min(num_diverse_tokens * 4, len(remaining_indices))
                sampled_idx = torch.randperm(len(remaining_indices), device=model_device)[:sample_size]
                sampled_indices = remaining_indices[sampled_idx]
                
                remaining_features = image_features[0, sampled_indices, :]
                remaining_features = F.normalize(remaining_features, p=2, dim=-1)
                
                diverse_idx = similarity_based_duplicate_removal_fast(
                    remaining_features, min(num_diverse_tokens, len(sampled_indices))
                )
                diverse_indices = sampled_indices[diverse_idx]
            else:
                # Full similarity computation for smaller sets
                remaining_features = image_features[0, remaining_indices, :]
                remaining_features = F.normalize(remaining_features, p=2, dim=-1)
                
                diverse_idx = similarity_based_duplicate_removal_fast(
                    remaining_features, num_diverse_tokens
                )
                diverse_indices = remaining_indices[diverse_idx]
    else:
        # Apply projection to all features if not done yet
        image_features = model.multi_modal_projector.to(model_device)(image_features)
    
    # Combine and sort indices in one operation
    selected_indices = torch.cat([important_indices, diverse_indices])
    selected_indices = torch.sort(selected_indices)[0]
    
    # Extract selected features
    image_features_selected = image_features[:, selected_indices, :]
    
    # Step 3: Optimized text-based recovery
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
                    if text_embeds.shape[-1] != image_features.shape[-1]:
                        # Use smaller projection layer
                        projection_layer = torch.nn.Linear(
                            text_embeds.shape[-1], image_features.shape[-1],
                            bias=False  # Remove bias for speed
                        ).to(model_device, dtype=torch.float16)
                        text_embeds = projection_layer(text_embeds)
                    
                    # Compute attention only for pruned tokens (memory efficient)
                    pruned_features = image_features[:, pruned_indices, :]
                    attention_scores = torch.einsum('btc,bpc->btp', text_embeds, pruned_features)
                    attention_scores = F.softmax(attention_scores, dim=-1).mean(dim=(0, 1))
                    
                    # Recover tokens
                    _, recovery_idx = torch.topk(attention_scores, num_tokens_to_recover)
                    recovery_indices = pruned_indices[recovery_idx]
                    
                    # Concatenate efficiently
                    image_features_recovered = image_features[:, recovery_indices, :]
                    image_features_selected = torch.cat(
                        [image_features_selected, image_features_recovered], dim=1
                    )
    
    # Single GPU-CPU transfer at the end
    result = image_features_selected.detach().cpu()
    
    print(f"Final output shape: {result.shape}")
    print(f"Kept {result.shape[1]} out of {N} tokens ({result.shape[1]/N*100:.1f}%)")
    
    return result


def similarity_based_duplicate_removal_fast(features, num_to_keep):
    """
    Optimized similarity-based token selection with approximate methods for speed.
    """
    n_tokens, dim = features.shape
    
    if n_tokens <= num_to_keep:
        return torch.arange(n_tokens, device=features.device)
    
    # For very large token sets, use clustering-based approximation
    if n_tokens > 1000:
        return cluster_based_selection(features, num_to_keep)
    
    # For medium-sized sets, use optimized greedy selection
    selected_idx = []
    selected_idx.append(0)  # Start with first token
    
    remaining_mask = torch.ones(n_tokens, dtype=torch.bool, device=features.device)
    remaining_mask[0] = False
    
    # Precompute all pairwise similarities (batch operation)
    similarity_matrix = torch.mm(features, features.t())
    
    for _ in range(num_to_keep - 1):
        remaining_indices = torch.nonzero(remaining_mask, as_tuple=True)[0]
        if len(remaining_indices) == 0:
            break
        
        # Find token with minimum maximum similarity to selected tokens
        selected_similarities = similarity_matrix[remaining_indices][:, selected_idx]
        max_similarities = selected_similarities.max(dim=1)[0]
        min_idx = max_similarities.argmin()
        
        next_token_idx = remaining_indices[min_idx].item()
        selected_idx.append(next_token_idx)
        remaining_mask[next_token_idx] = False
    
    return torch.tensor(selected_idx, device=features.device)


def cluster_based_selection(features, num_to_keep):
    """
    Fast approximate selection using k-means clustering.
    """
    from sklearn.cluster import MiniBatchKMeans
    import numpy as np
    
    # Convert to numpy for sklearn
    features_np = features.detach().cpu().numpy()
    
    # Use more clusters than needed, then select representatives
    n_clusters = min(num_to_keep * 2, len(features_np))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)
    cluster_labels = kmeans.fit_predict(features_np)
    
    # Select one representative per cluster, prioritizing diverse clusters
    selected_idx = []
    cluster_centers = kmeans.cluster_centers_
    
    # Calculate inter-cluster distances
    cluster_distances = np.linalg.norm(
        cluster_centers[:, np.newaxis] - cluster_centers[np.newaxis, :], axis=2
    )
    
    # Greedy selection of diverse clusters
    selected_clusters = []
    remaining_clusters = list(range(n_clusters))
    
    # Start with cluster closest to mean
    mean_features = features_np.mean(axis=0)
    distances_to_mean = np.linalg.norm(cluster_centers - mean_features, axis=1)
    first_cluster = distances_to_mean.argmin()
    selected_clusters.append(first_cluster)
    remaining_clusters.remove(first_cluster)
    
    while len(selected_clusters) < num_to_keep and remaining_clusters:
        # Find cluster most distant from selected ones
        min_distances = []
        for cluster in remaining_clusters:
            min_dist = min(cluster_distances[cluster][selected] for selected in selected_clusters)
            min_distances.append(min_dist)
        
        next_cluster = remaining_clusters[np.argmax(min_distances)]
        selected_clusters.append(next_cluster)
        remaining_clusters.remove(next_cluster)
    
    # Select one representative token from each selected cluster
    for cluster_id in selected_clusters:
        cluster_tokens = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_tokens) > 0:
            # Select token closest to cluster center
            cluster_features = features_np[cluster_tokens]
            distances = np.linalg.norm(cluster_features - cluster_centers[cluster_id], axis=1)
            representative = cluster_tokens[distances.argmin()]
            selected_idx.append(representative)
    
    return torch.tensor(selected_idx[:num_to_keep], device=features.device)


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
        {'keep_ratio': 0.5, 'recovery_ratio': 0.05},
        {'keep_ratio': 0.5, 'recovery_ratio': 0.1},
        {'keep_ratio': 0.5, 'recovery_ratio': 0.2},
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
