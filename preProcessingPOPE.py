import pandas as pd
import os
from pathlib import Path
from PIL import Image
import io

def extract_images_from_parquet(parquet_file='POPE.parquet', output_dir='POPE'):
    """
    Extract images from POPE.parquet file and save them using image_source column as filenames
    
    Args:
        parquet_file (str): Path to the parquet file
        output_dir (str): Directory to save extracted images
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Read the parquet file
        print(f"Reading {parquet_file}...")
        df = pd.read_parquet(parquet_file)
        
        print(f"Found {len(df)} rows in the dataset")
        print(f"Columns: {list(df.columns)}")
        
        # Check if required columns exist
        if 'image' not in df.columns:
            raise ValueError("'image' column not found in the parquet file")
        if 'image_source' not in df.columns:
            raise ValueError("'image_source' column not found in the parquet file")
        
        successful_extractions = 0
        failed_extractions = 0
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                # Get image binary data and source filename
                image_binary = row['image']
                image_source = row['image_source']
                
                # Skip if either is None/empty
                if image_binary is None or image_source is None:
                    print(f"Row {idx}: Skipping - missing image data or source name")
                    failed_extractions += 1
                    continue
                
                # Clean the filename (remove any path separators and invalid characters)
                safe_filename = str(image_source).replace('/', '_').replace('\\', '_')
                
                # If the filename doesn't have an extension, try to detect it or add .jpg as default
                if not any(safe_filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']):
                    # Try to detect format from binary data
                    try:
                        img = Image.open(io.BytesIO(image_binary))
                        format_ext = img.format.lower() if img.format else 'jpg'
                        if format_ext == 'jpeg':
                            format_ext = 'jpg'
                        safe_filename += f'.{format_ext}'
                    except:
                        safe_filename += '.jpg'  # Default extension
                
                # Create full output path
                output_path = os.path.join(output_dir, safe_filename)
                
                # Convert binary data to image and save
                image = Image.open(io.BytesIO(image_binary))
                image.save(output_path)
                
                successful_extractions += 1
                if successful_extractions % 100 == 0:  # Progress update every 100 images
                    print(f"Processed {successful_extractions} images...")
                
            except Exception as e:
                print(f"Row {idx}: Failed to extract image '{image_source}' - {str(e)}")
                failed_extractions += 1
                continue
        
        print(f"\nExtraction complete!")
        print(f"Successfully extracted: {successful_extractions} images")
        print(f"Failed extractions: {failed_extractions}")
        print(f"Images saved to: {os.path.abspath(output_dir)}")
        
    except FileNotFoundError:
        print(f"Error: Could not find {parquet_file}")
        print("Make sure the file exists in the current directory")
    except Exception as e:
        print(f"Error reading parquet file: {str(e)}")

if __name__ == "__main__":
    # Run the extraction
    extract_images_from_parquet()
