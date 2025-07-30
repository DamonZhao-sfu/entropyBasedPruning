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


def compute_pruning_scores(attention_matrix, lambda_val, alpha=0.5):
    """
    Compute pruning scores with mutual information approximation.
    
    Args:
        attention_matrix: [num_text_tokens, num_image_tokens] attention matrix
        lambda_val: Hyperparameter for entropy weight
        alpha: Weight for mutual information term
    
    Returns:
        scores: Comprehensive score for each image token
    """
    I = attention_matrix.sum(dim=0)  # Importance score [num_image_tokens]
    attention_per_image_token = attention_matrix / (attention_matrix.sum(dim=0, keepdim=True) + 1e-10)
    attention_per_image_token = torch.clamp(attention_per_image_token, min=1e-10)
    H = - (attention_per_image_token * torch.log(attention_per_image_token)).sum(dim=0)  # Entropy
    
    # Approximate mutual information: higher variance in attention indicates higher unique contribution
    attention_variance = torch.var(attention_per_image_token, dim=0)  # [num_image_tokens]
    mi_approx = attention_variance / (attention_variance.max() + 1e-10)  # Normalize
    
    scores = alpha * I + (1 - alpha) * mi_approx - lambda_val * H
    return scores


from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
def getPrunedVisualToken(model, image_path, texts, keep_ratio=0.125, lambda_val=0.1, num_clusters=4):
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
    
    # Project image features
    B, N, C = image_features.shape
    image_features = image_features.to(device=model_device, dtype=torch.float16)
    model.multi_modal_projector = model.multi_modal_projector.to(model_device)
    image_features = model.multi_modal_projector(image_features)
    
    # Get projected dimensions
    projected_dim = image_features.shape[-1]  # This should be 4096
    
    # Compute text-guided attention matrix using text_embeds and image_features
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
        
        attention_matrix = F.softmax(attention_matrix, dim=-1)  # Normalize over image tokens
        attention_matrix = attention_matrix.mean(dim=0)  # Average over batch: [num_text_tokens, num_image_tokens]
    
    # Compute pruning scores
    scores = compute_pruning_scores(attention_matrix, lambda_val)
    
    # # Optimization: Cluster-based token selection
    attention_per_image = attention_matrix / (attention_matrix.sum(dim=0, keepdim=True) + 1e-10)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    # cluster_labels = torch.tensor(kmeans.fit_predict(attention_per_image.t().detach().cpu().numpy()), device=model_device)
    
    attention_data = attention_per_image.t().detach().cpu().numpy()
    
    # Apply SimpleImputer to handle NaN values
    imputer = SimpleImputer(strategy='mean')  # You can also use 'median' or 'constant'
    attention_data_imputed = imputer.fit_transform(attention_data)
    
    # Convert back to tensor and move to model device
    cluster_labels = torch.tensor(kmeans.fit_predict(attention_data_imputed), device=model_device)

    num_image_tokens = N
    num_tokens_to_keep = min(int(keep_ratio * num_image_tokens), num_image_tokens)
    top_indices = []
    tokens_per_cluster = max(1, num_tokens_to_keep // num_clusters)
    
    for cluster_id in range(num_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_scores = scores[cluster_mask]
        cluster_indices = torch.arange(num_image_tokens, device=model_device)[cluster_mask]
        if cluster_scores.numel() > 0:
            _, top_cluster_indices = torch.topk(cluster_scores, min(tokens_per_cluster, cluster_scores.numel()), dim=-1)
            top_indices.append(cluster_indices[top_cluster_indices])
    
    top_indices = torch.cat(top_indices)[:num_tokens_to_keep]  # Ensure exact number of tokens
    if (top_indices >= num_image_tokens).any() or (top_indices < 0).any():
        raise ValueError(f"top_indices contains invalid values: {top_indices}")
    
    image_features_selected = image_features[:, top_indices, :]  # [B, num_tokens_to_keep, C]
    image_features_selected = image_features_selected.detach().cpu()
    print(f"Final output shape: {image_features_selected.shape}")
    return image_features_selected

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

if __name__ == "__main__":
    MODEL_PATH = "/data/models/llava-1.5-7b-hf"
    IMAGE_PATH = "/home/haikai/MMbench/extracted_images/0.jpg"
    TEXTS = "Describe the main object in the image"
    API_URL = "http://localhost:8005"

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="cuda",
        attn_implementation="eager"
    )

    processor = LlavaProcessor.from_pretrained(MODEL_PATH, patch_size=14)
    print("Comparing different token pruning methods...")    
    # Initialize results storage
    results_data = []
    
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
    ).limit(1000).collect()
    
    print(f"Testing {len(sample_data)} questions...")
    print("-" * 60)

    # Run tests with CSV logging
    for i, row in enumerate(sample_data, 1):
        question = row['question']  if row['question'] else ""
        hint = row['hint'] if row['hint'] else ""
        correct_answer = row['answer']  if row['answer']  else ""
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
            'original_token': 0,
            'pruned_token': 0,
            'api_url': API_URL,
            'method': 'embeddings'
        }
        
        try:
            if True:
                # Embedding method - use existing pruning logic
                prune_time_begin = time.time()
                reduced_tokens = getPrunedVisualToken(model, image_path, formatted_question)
                #reduced_tokens, originaTokenNum, preprocess_time, encode_time, project_time, pruneTime = divprune(model, processor, image_path, device="cuda", prune=False)
                prune_time_end = time.time()

                embed_time = prune_time_end - prune_time_begin
                result_record['original_token'] = 576
                result_record['pruned_token'] = reduced_tokens.shape[1]
                result_record['preprocess_time'] = 0
                result_record['encode_time'] = 0
                result_record['project_time'] = 0
                result_record['prune_time'] = 0

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
            result_record['embed_time'] = embed_time
            result_record['api_call_time'] = api_call_time
            result_record['total_time'] = embed_time + api_call_time
            
            if response:
                result_record['api_success'] = True
                print(f"embed time: {embed_time:.2f} seconds")
                print(f"api call time: {api_call_time:.2f} seconds")
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
        
        # Add the result to our data list
        results_data.append(result_record)
        print()


    # Save results to CSV
    csv_filename = f"api_timing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_results_to_csv(results_data, csv_filename)
    
    # Generate and display timing summary
    generate_timing_summary(results_data)
     # Calculate accuracy
    successful_tests = [r for r in results_data if r['api_success'] and r['predicted_answer']]
    if successful_tests:
        accuracy = sum(1 for r in successful_tests if r['is_correct']) / len(successful_tests)
        print(f"\nOverall Accuracy: {accuracy:.2%} ({sum(1 for r in successful_tests if r['is_correct'])}/{len(successful_tests)})")



    #pruned_features = getPrunedVisualToken(model, IMAGE_PATH, TEXTS, keep_ratio=0.125, lambda_val=0.1)