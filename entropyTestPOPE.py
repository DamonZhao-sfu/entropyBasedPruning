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
        "guided_choice": ["yes", "no"]  # For POPE yes/no questions
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


def similarity_based_duplicate_removal(normalized_tokens, target_count, batch_size=2):
    """
    Implementation of Algorithm 1 from VisPruner paper
    Removes duplicate tokens based on similarity until target_count tokens remain
    
    Args:
        normalized_tokens: L2-normalized token features [N, C]
        target_count: desired number of diverse tokens
        batch_size: number of tokens to remove per iteration
    
    Returns:
        remaining_idx: indices of diverse tokens
    """
    N, C = normalized_tokens.shape
    remaining_idx = torch.arange(N, device=normalized_tokens.device)
    
    while len(remaining_idx) > target_count:
        # Number to remove this iteration
        r = min(batch_size, len(remaining_idx) - target_count)
        if r <= 0:
            break
            
        remaining = normalized_tokens[remaining_idx]
        
        # Split into two groups for pairwise similarity computation
        a = remaining[::2]  # even indices
        b = remaining[1::2]  # odd indices
        
        # Compute cosine similarity matrix
        score = torch.mm(a, b.transpose(-1, -2))  # [len(a), len(b)]
        
        # Find maximum similarity for each token in group a
        max_scores = score.max(dim=-1).values  # [len(a)]
        
        # Sort by similarity (descending) and keep the least similar ones
        _, sorted_indices = max_scores.sort(descending=True)
        diverse_idx = sorted_indices[r:]  # remove r most similar tokens
        
        # Update remaining indices (keep diverse tokens from even group + all odd group)
        even_remaining = remaining_idx[::2][diverse_idx]
        odd_remaining = remaining_idx[1::2]
        remaining_idx = torch.cat([even_remaining, odd_remaining], dim=0)
    
    return remaining_idx

def getPrunedVisualTokenVisPruner(model, image_binary, texts, keep_ratio=0.125, 
                                  important_ratio=0.6, recovery_ratio=0.1):
    """
    Adapted version using VisPruner approach:
    1. Use visual attention ([CLS] attention) to select important tokens
    2. Use similarity-based pruning to select diverse tokens from remaining ones
    3. Optional: recover some tokens based on text relevance
    
    Args:
        model: vision-language model
        image_binary: binary image data
        texts: text input (can be None)
        keep_ratio: total ratio of tokens to keep
        important_ratio: ratio of kept tokens that should be "important" (rest will be "diverse")
        recovery_ratio: ratio of additional tokens to recover based on text relevance
    """
    # Convert binary data to PIL Image
    image = Image.open(io.BytesIO(image_binary))
    inputs = vision_tower.image_processor(image, return_tensors="pt")
    images = inputs["pixel_values"]
    
    model_device = vision_tower.device
    
    # Process image features and get attention from visual encoder
    with torch.no_grad():
        image_forward_outs = vision_tower.vision_tower(
            images.to(device=model_device, dtype=vision_tower.dtype),
            output_hidden_states=True,
            output_attentions=True
        )
        
        # Extract [CLS] attention from the visual encoder (penultimate layer)
        # This is the key insight from VisPruner paper
        attentions = image_forward_outs.attentions
        if len(attentions) > 1:
            # Use penultimate layer attention as suggested in the paper
            cls_attention = attentions[-2][0, :, 0, 1:].mean(dim=0)  # Average across heads, [CLS] -> patches
        else:
            cls_attention = attentions[-1][0, :, 0, 1:].mean(dim=0)
        
        image_outputs = vision_tower.feature_select(image_forward_outs)
        image_features = image_outputs.to(images.dtype)
    
    # Process text embeddings (if provided)
    text_embeds = None
    if texts is not None:
        with torch.no_grad():
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
                text_embeds = text_embeds.unsqueeze(0)
            elif text_embeds.dim() > 3:
                text_embeds = text_embeds.squeeze().reshape(1, -1, text_embeds.size(-1))
    
    B, N, C = image_features.shape
    image_features = image_features.to(device=model_device, dtype=torch.float16)
    model.multi_modal_projector = model.multi_modal_projector.to(model_device)
    image_features = model.multi_modal_projector(image_features)
    
    # Calculate how many tokens to keep in each category
    num_tokens_to_keep = min(int(keep_ratio * N), N)
    num_important_tokens = int(num_tokens_to_keep * important_ratio)
    num_diverse_tokens = num_tokens_to_keep - num_important_tokens
    
    # Step 1: Select important tokens based on [CLS] attention (VisPruner approach)
    cls_attention = cls_attention.to(model_device)
    _, important_indices = torch.topk(cls_attention, num_important_tokens, dim=-1)
    
    # Step 2: Select diverse tokens from remaining ones using similarity-based pruning
    all_indices = torch.arange(N, device=model_device)
    remaining_indices = all_indices[~torch.isin(all_indices, important_indices)]
    
    if num_diverse_tokens > 0 and len(remaining_indices) > 0:
        # Normalize remaining tokens for similarity computation
        remaining_features = image_features[0, remaining_indices, :]  # [remaining_N, C]
        remaining_features_norm = F.normalize(remaining_features, p=2, dim=-1)
        
        # Apply similarity-based duplicate removal
        if len(remaining_indices) > num_diverse_tokens:
            diverse_idx = similarity_based_duplicate_removal(
                remaining_features_norm, num_diverse_tokens
            )
            diverse_indices = remaining_indices[diverse_idx]
        else:
            diverse_indices = remaining_indices
    else:
        diverse_indices = torch.tensor([], device=model_device, dtype=torch.long)
    
    # Combine important and diverse tokens
    selected_indices = torch.cat([important_indices, diverse_indices], dim=0)
    
    # Sort indices to maintain original order
    selected_indices, _ = torch.sort(selected_indices)
    
    # Extract selected features
    image_features_selected = image_features[:, selected_indices, :]
    
    # Step 3: Optional recovery based on text relevance (your original approach)
    if texts is not None and recovery_ratio > 0:
        text_embeds = text_embeds.to(device=model_device, dtype=torch.float16)
        
        # Get indices of pruned tokens
        pruned_indices = all_indices[~torch.isin(all_indices, selected_indices)]
        
        if len(pruned_indices) > 0:
            # Compute text-visual attention for recovery
            if text_embeds.shape[-1] != image_features.shape[-1]:
                projection_layer = torch.nn.Linear(
                    text_embeds.shape[-1], image_features.shape[-1]
                ).to(model_device).half()
                text_embeds_projected = projection_layer(text_embeds)
                attention_matrix = torch.bmm(
                    text_embeds_projected, image_features.transpose(1, 2)
                )
            else:
                attention_matrix = torch.bmm(text_embeds, image_features.transpose(1, 2))
            
            attention_matrix = F.softmax(attention_matrix, dim=-1)
            attention_matrix = attention_matrix.mean(dim=0)
            
            # Recover additional tokens based on text relevance
            num_tokens_to_recover = min(
                int(recovery_ratio * N), len(pruned_indices)
            )
            if num_tokens_to_recover > 0:
                recovery_scores = attention_matrix.sum(dim=0)[pruned_indices]
                _, recovery_idx = torch.topk(recovery_scores, num_tokens_to_recover, dim=-1)
                recovery_indices = pruned_indices[recovery_idx]
                
                image_features_recovered = image_features[:, recovery_indices, :]
                image_features_selected = torch.cat(
                    [image_features_selected, image_features_recovered], dim=1
                )
    
    image_features_selected = image_features_selected.detach().cpu()
    print(f"Final output shape: {image_features_selected.shape}")
    print(f"Kept {image_features_selected.shape[1]} out of {N} tokens "
          f"({image_features_selected.shape[1]/N*100:.1f}%)")
    
    return image_features_selected


def getPrunedVisualToken(model, image_binary, texts, keep_ratio=0.125, lambda_val=0.1, recovery_ratio=0.1):
    """
    Process image from binary data instead of file path
    """
    # Convert binary data to PIL Image
    image = Image.open(io.BytesIO(image_binary))
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
                text_embeds = text_embeds.unsqueeze(0)
            elif text_embeds.dim() > 3:
                text_embeds = text_embeds.squeeze().reshape(1, -1, text_embeds.size(-1))
    
    torch.cuda.synchronize()
    
    B, N, C = image_features.shape
    image_features = image_features.to(device=model_device, dtype=torch.float16)
    model.multi_modal_projector = model.multi_modal_projector.to(model_device)
    image_features = model.multi_modal_projector(image_features)
    
    if texts is not None:
        text_embeds = text_embeds.to(device=model_device, dtype=torch.float16)

        if text_embeds.shape[-1] != image_features.shape[-1]:
            projection_layer = torch.nn.Linear(text_embeds.shape[-1], image_features.shape[-1]).to(model_device).half()
            text_embeds_projected = projection_layer(text_embeds)
            attention_matrix = torch.bmm(text_embeds_projected, image_features.transpose(1, 2))
        else:
            attention_matrix = torch.bmm(text_embeds, image_features.transpose(1, 2))
        
        attention_matrix = F.softmax(attention_matrix, dim=-1)  
        attention_matrix = attention_matrix.mean(dim=0)  
        
    # Compute pruning scores
    eps = 1e-10
    attention_sum = attention_matrix.sum(dim=0, keepdim=True)
    attention_per_image_token = attention_matrix / (attention_sum + eps)
    attention_per_image_token = torch.clamp(attention_per_image_token, min=eps)
    
    log_attention = torch.log(attention_per_image_token)
    H = -(attention_per_image_token * log_attention).sum(dim=0)
    
    scores = -lambda_val * H
    
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
        summary_token = image_features[:, pruned_indices, :].mean(dim=1, keepdim=True)
        image_features_selected = torch.cat(
            [image_features[:, top_indices, :], summary_token], dim=1
        )
    else:
        image_features_selected = image_features[:, top_indices, :]
    
    # Recover additional tokens based on text relevance
    num_tokens_to_recover = min(int(recovery_ratio * num_image_tokens), pruned_indices.numel())
    if num_tokens_to_recover > 0:
        recovery_scores = attention_matrix.sum(dim=0)[pruned_indices]
        _, recovery_indices = torch.topk(recovery_scores, num_tokens_to_recover, dim=-1)
        recovery_indices = pruned_indices[recovery_indices]
        image_features_recovered = image_features[:, recovery_indices, :]
        image_features_selected = torch.cat(
            [image_features_selected, image_features_recovered], dim=1
        )
    
    image_features_selected = image_features_selected.detach().cpu()
    print(f"Final output shape: {image_features_selected.shape}")
    
    return image_features_selected


def getPrunedVisualTokenVisPruner_optimized(model, image_binary, texts, keep_ratio=0.125, 
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
    image = Image.open(io.BytesIO(image_binary))
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



def extract_image_binary_from_pope_data(image_data):
    """
    Extract binary data from POPE dataset image column
    """
    if isinstance(image_data, dict):
        # Look for bytes in dictionary
        if 'bytes' in image_data:
            return image_data['bytes']
        elif 'data' in image_data:
            return image_data['data'] 
        elif 'binary' in image_data:
            return image_data['binary']
    elif hasattr(image_data, 'bytes'):
        return image_data.bytes
    elif isinstance(image_data, bytes):
        return image_data
    elif hasattr(image_data, 'save'):
        # PIL Image object
        buffer = io.BytesIO()
        image_data.save(buffer, format='JPEG')
        return buffer.getvalue()
    
    raise ValueError(f"Could not extract binary data from image_data of type {type(image_data)}")

def save_results_to_csv(results_data, filename="pope_results.csv"):
    """
    Save POPE query results to CSV file
    """
    if not results_data:
        print("No results data to save")
        return
    
    headers = [
        'question_id',
        'question',
        'correct_answer',
        'predicted_answer',
        'is_correct',
        'image_source',
        'category',
        'keep_ratio',
        'recovery_ratio',
        'embed_time',
        'api_call_time', 
        'total_time',
        'api_success',
        'generated_text',
        'error_message',
        'original_tokens',
        'pruned_tokens',
        'timestamp'
    ]
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            
            for row in results_data:
                complete_row = {header: row.get(header, '') for header in headers}
                writer.writerow(complete_row)
        
        print(f"Results saved to {filename}")
        print(f"Number of records saved: {len(results_data)}")
        
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        import traceback
        traceback.print_exc()

def save_config_results_separately(all_results, base_filename="pope_config_results"):
    """
    Save results for each configuration to separate CSV files
    """
    if not all_results:
        print("No results data to save")
        return
    
    # Group results by configuration
    config_groups = {}
    for result in all_results:
        config_key = f"keep_{result['keep_ratio']}_recovery_{result['recovery_ratio']}"
        if config_key not in config_groups:
            config_groups[config_key] = []
        config_groups[config_key].append(result)
    
    # Save each configuration to a separate file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_files = []
    
    for config_key, results in config_groups.items():
        filename = f"{base_filename}_{config_key}_{timestamp}.csv"
        save_results_to_csv(results, filename)
        saved_files.append(filename)
        
        # Print brief summary for this config
        successful_tests = [r for r in results if r['api_success']]
        if successful_tests:
            total_correct = sum(1 for r in successful_tests if r['is_correct'])
            accuracy = total_correct / len(successful_tests) if successful_tests else 0
            avg_tokens = sum(r['pruned_tokens'] for r in successful_tests) / len(successful_tests)
            print(f"  {config_key}: {len(results)} samples, {accuracy:.2%} accuracy, {avg_tokens:.1f} avg tokens")
    
    return saved_files

def print_configuration_summary(all_results):
    """
    Print summary statistics for each configuration
    """
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    
    # Group results by configuration
    config_groups = {}
    for result in all_results:
        if result['api_success']:
            config_key = f"keep_{result['keep_ratio']}_recovery_{result['recovery_ratio']}"
            if config_key not in config_groups:
                config_groups[config_key] = []
            config_groups[config_key].append(result)
    
    # Print summary for each configuration
    for config_key, results in config_groups.items():
        if results:
            total_correct = sum(1 for r in results if r['is_correct'])
            accuracy = total_correct / len(results)
            avg_embed_time = sum(r['embed_time'] for r in results) / len(results)
            avg_api_time = sum(r['api_call_time'] for r in results) / len(results)
            avg_pruned_tokens = sum(r['pruned_tokens'] for r in results) / len(results)
            token_reduction = ((576 - avg_pruned_tokens) / 576 * 100)
            
            print(f"\n{config_key.upper().replace('_', ' ')}:")
            print(f"  Samples: {len(results)}")
            print(f"  Accuracy: {accuracy:.2%} ({total_correct}/{len(results)})")
            print(f"  Avg Embed Time: {avg_embed_time:.2f}s")
            print(f"  Avg API Time: {avg_api_time:.2f}s") 
            print(f"  Avg Pruned Tokens: {avg_pruned_tokens:.1f}")
            print(f"  Token Reduction: {token_reduction:.1f}%")

if __name__ == "__main__":
    MODEL_PATH = "/data/models/llava-1.5-7b-hf"
    POPE_PARQUET_PATH = "POPE.parquet"  # Update this path
    API_URL = "http://localhost:8005"

    # Define different configurations to test
    PRUNING_CONFIGS = [
        #{'keep_ratio': 0.05, 'recovery_ratio': 0.05},
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

    # Load the model
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="cuda",
        attn_implementation="eager"
    )

    processor = LlavaProcessor.from_pretrained(MODEL_PATH, patch_size=14)
    
    print("Loading POPE dataset...")
    
    try:
        # Read POPE parquet file
        df = pd.read_parquet(POPE_PARQUET_PATH)
        print(f"Loaded {len(df)} samples from POPE dataset")
        print(f"Columns: {list(df.columns)}")
        
        # Initialize results storage
        all_results = []
        
        # Sample first subset for testing (remove .head() to process all)
        sample_data = df
        
        print(f"Processing {len(sample_data)} questions with {len(PRUNING_CONFIGS)} configurations...")
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
            
            config_results = []
            
            # Process all samples with current configuration
            for idx, row in sample_data.iterrows():
                question_id = row.get('question_id', f'q_{idx}')
                question = row.get('question', '')
                correct_answer = row.get('answer', '').lower().strip()
                image_source = row.get('image_source', f'image_{idx}')
                category = row.get('category', 'unknown')
                image_data = row.get('image')
                
                print(f"\nSample {idx+1}/{len(sample_data)}: {question_id}")
                print(f"Question: {question}")
                print(f"Correct Answer: {correct_answer}")
                
                # Initialize result record for this configuration
                result_record = {
                    'question_id': question_id,
                    'question': question,
                    'correct_answer': correct_answer,
                    'image_source': image_source,
                    'category': category,
                    'keep_ratio': keep_ratio,
                    'recovery_ratio': recovery_ratio,
                    'predicted_answer': '',
                    'is_correct': False,
                    'embed_time': 0,
                    'api_call_time': 0,
                    'total_time': 0,
                    'api_success': False,
                    'generated_text': '',
                    'error_message': '',
                    'original_tokens': 576,  # Standard CLIP vision tokens
                    'pruned_tokens': 0,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                try:
                    # Extract image binary data
                    image_binary = extract_image_binary_from_pope_data(image_data)
                    
                    # Process with current pruning configuration
                    embed_start = time.time()

                    reduced_tokens = getPrunedVisualTokenVisPruner_optimized(
                        model,
                        image_binary,
                        question,
                        keep_ratio=keep_ratio,
                        important_ratio=0.6,
                        recovery_ratio=recovery_ratio
                    )
                    # reduced_tokens = getPrunedVisualToken(
                    #     model, 
                    #     image_binary, 
                    #     question,
                    #     keep_ratio=keep_ratio,
                    #     lambda_val=0.1,
                    #     recovery_ratio=recovery_ratio
                    # )
                    embed_end = time.time()
                    
                    embed_time = embed_end - embed_start
                    result_record['embed_time'] = embed_time
                    result_record['pruned_tokens'] = reduced_tokens.shape[1]
                    
                    # Call API
                    api_start = time.time()
                    response = call_vllm_api_with_embeds(
                        image_embedding=reduced_tokens.to(torch.float16),
                        question=question,
                        model="/data/models/llava-1.5-7b-hf",
                        api_url=API_URL
                    )
                    api_end = time.time()
                    
                    api_call_time = api_end - api_start
                    result_record['api_call_time'] = api_call_time
                    result_record['total_time'] = embed_time + api_call_time
                    
                    if response and 'choices' in response and len(response['choices']) > 0:
                        result_record['api_success'] = True
                        content = response['choices'][0]['message']['content']
                        result_record['generated_text'] = content
                        
                        # Extract yes/no answer
                        content_lower = content.lower().strip()
                        if 'yes' in content_lower:
                            predicted_answer = 'yes'
                        elif 'no' in content_lower:
                            predicted_answer = 'no'
                        else:
                            predicted_answer = content_lower
                        
                        result_record['predicted_answer'] = predicted_answer
                        result_record['is_correct'] = (predicted_answer == correct_answer)
                        
                        print(f"Generated: {content}")
                        print(f"Predicted: {predicted_answer}")
                        print(f"Correct: {result_record['is_correct']}")
                        print(f"Tokens: {result_record['pruned_tokens']}")
                        print(f"Times - Embed: {embed_time:.2f}s, API: {api_call_time:.2f}s")
                        
                    else:
                        result_record['error_message'] = "No valid response from API"
                        print(f"No valid response from API")
                        
                except Exception as e:
                    result_record['error_message'] = str(e)
                    print(f"Error processing sample {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                
                config_results.append(result_record)
                all_results.append(result_record)
                print("-" * 40)
            
            # Save results for this configuration immediately
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            config_filename = f"pope_results_keep{keep_ratio}_recovery{recovery_ratio}_{timestamp}.csv"
            save_results_to_csv(config_results, config_filename)
            
            # Print summary for this configuration
            successful_tests = [r for r in config_results if r['api_success']]
            if successful_tests:
                total_correct = sum(1 for r in successful_tests if r['is_correct'])
                accuracy = total_correct / len(successful_tests)
                avg_embed_time = sum(r['embed_time'] for r in successful_tests) / len(successful_tests)
                avg_api_time = sum(r['api_call_time'] for r in successful_tests) / len(successful_tests)
                avg_pruned_tokens = sum(r['pruned_tokens'] for r in successful_tests) / len(successful_tests)
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
        
        # Save combined results file (optional)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        combined_csv_filename = f"pope_all_configs_combined_{timestamp}.csv"
        save_results_to_csv(all_results, combined_csv_filename)
        
        # Print final configuration summary
        print_configuration_summary(all_results)
        
        # Print overall statistics
        successful_tests = [r for r in all_results if r['api_success']]
        if successful_tests:
            print(f"\n" + "="*80)
            print("OVERALL STATISTICS")
            print("="*80)
            print(f"Total Samples: {len(sample_data)}")
            print(f"Total Configurations: {len(PRUNING_CONFIGS)}")
            print(f"Total Evaluations: {len(all_results)}")
            print(f"Successful API Calls: {len(successful_tests)}")
            print(f"Success Rate: {len(successful_tests)/len(all_results):.2%}")
        
    except Exception as e:
        print(f"Error loading POPE dataset: {e}")
        import traceback
        traceback.print_exc()
