#!/usr/bin/env python3
"""
Generate training data from parsed text files using vLLM
Replacement for synthetic-data-kit's broken generation step
"""

import json
import yaml
import requests
from pathlib import Path
from tqdm import tqdm
import argparse


def load_config(config_path):
    """Load YAML config"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_text_file(file_path):
    """Load and chunk a text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def chunk_text(text, chunk_size=4000, overlap=300):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.7:  # Only if we're not cutting too much
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


def generate_qa_pairs(text_chunk, config, num_pairs=5):
    """Generate QA pairs from a text chunk using vLLM"""
    
    vllm_config = config['vllm']
    api_base = vllm_config['api_base']
    model = vllm_config['model']
    
    # Get the prompt template from config
    prompt_template = config['prompts']['qa_generation']
    prompt = prompt_template.format(num_pairs=num_pairs, text=text_chunk)
    
    # Call vLLM API
    try:
        response = requests.post(
            f"{api_base}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates educational question-answer pairs. You ALWAYS output valid JSON arrays with no markdown formatting."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": config['generation']['temperature'],
                "max_tokens": config['generation']['max_tokens'],
                "top_p": config['generation']['top_p']
            },
            headers={"Content-Type": "application/json"},
            timeout=60
        )
    except requests.exceptions.RequestException as e:
        print(f"\n{'='*60}")
        print(f"❌ API REQUEST FAILED")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print(f"API Base: {api_base}")
        print(f"Model: {model}")
        print(f"{'='*60}\n")
        raise
    
    # Check response status
    if response.status_code != 200:
        print(f"\n{'='*60}")
        print(f"❌ API RETURNED ERROR STATUS: {response.status_code}")
        print(f"{'='*60}")
        print(f"Response: {response.text[:500]}")
        print(f"{'='*60}\n")
        raise Exception(f"API request failed: {response.status_code}")
    
    # Parse response
    try:
        result = response.json()
    except json.JSONDecodeError:
        print(f"\n{'='*60}")
        print(f"❌ RESPONSE IS NOT VALID JSON")
        print(f"{'='*60}")
        print(f"Raw response text: {response.text[:500]}")
        print(f"{'='*60}\n")
        raise
    
    # Extract content
    content = result['choices'][0]['message']['content']
    
    # DEBUG: Print first chunk's response
    print(f"\n{'='*60}")
    print("📝 RAW MODEL OUTPUT (first 500 chars):")
    print(f"{'='*60}")
    print(content[:500])
    print(f"{'='*60}\n")
    
    # Clean the response (remove markdown if present)
    content = content.strip()
    if content.startswith('```json'):
        content = content[7:]  # Remove ```json
    if content.startswith('```'):
        content = content[3:]  # Remove ```
    if content.endswith('```'):
        content = content[:-3]  # Remove ```
    content = content.strip()
    
    # Parse JSON
    try:
        qa_pairs = json.loads(content)
        return qa_pairs if isinstance(qa_pairs, list) else [qa_pairs]
    except json.JSONDecodeError as e:
        print(f"\n{'='*60}")
        print("FAILED TO PARSE JSON")
        print(f"{'='*60}")
        print("RAW OUTPUT:")
        print(content[:500])  # First 500 chars
        print(f"\nError: {e}")
        print(f"{'='*60}\n")
        
        # Try to extract JSON if it's wrapped in text
        import re
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            try:
                qa_pairs = json.loads(json_match.group(0))
                print("✓ Recovered JSON from wrapped text!")
                return qa_pairs if isinstance(qa_pairs, list) else [qa_pairs]
            except:
                pass
        
        return []


def process_file(file_path, config, num_pairs_per_chunk=5):
    """Process a single text file and generate QA pairs"""
    
    print(f"\n📄 Processing: {file_path.name}")
    
    # Load and chunk the text
    text = load_text_file(file_path)
    chunks = chunk_text(
        text, 
        chunk_size=config['generation']['chunk_size'],
        overlap=config['generation']['overlap']
    )
    
    print(f"   Split into {len(chunks)} chunks")
    
    all_qa_pairs = []
    
    # Process each chunk
    for i, chunk in enumerate(tqdm(chunks, desc=f"   Generating from chunks")):
        try:
            qa_pairs = generate_qa_pairs(chunk, config, num_pairs=num_pairs_per_chunk)
            all_qa_pairs.extend(qa_pairs)
        except Exception as e:
            import traceback
            print(f"\n   ⚠️  Chunk {i} failed")
            print(f"   Exception type: {type(e).__name__}")
            print(f"   Exception message: {str(e)}")
            print(f"   Traceback:")
            traceback.print_exc()
            continue
    
    print(f"   ✓ Generated {len(all_qa_pairs)} QA pairs")
    
    return all_qa_pairs


def save_qa_pairs(qa_pairs, output_path):
    """Save QA pairs to JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved to: {output_path}")


def convert_to_alpaca_format(qa_pairs):
    """Convert QA pairs to Alpaca training format"""
    
    alpaca_data = []
    
    for qa in qa_pairs:
        alpaca_data.append({
            "instruction": qa.get("question", ""),
            "input": "",
            "output": qa.get("answer", ""),
            "domain": qa.get("domain", "unknown"),
            "difficulty": qa.get("difficulty", "intermediate")
        })
    
    return alpaca_data


def main():
    parser = argparse.ArgumentParser(description="Generate training data from parsed text files")
    parser.add_argument("--config", "-c", required=True, help="Path to config YAML file")
    parser.add_argument("--input-dir", "-i", required=True, help="Directory with parsed .txt files")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for generated QA pairs")
    parser.add_argument("--num-pairs", "-n", type=int, default=5, help="Number of QA pairs per chunk")
    parser.add_argument("--format", choices=['raw', 'alpaca'], default='raw', help="Output format")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get input files
    input_dir = Path(args.input_dir)
    txt_files = list(input_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"❌ No .txt files found in {input_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"📚 Found {len(txt_files)} text files to process")
    print(f"🤖 Using model: {config['vllm']['model']}")
    print(f"🎯 Target: {args.num_pairs} QA pairs per chunk")
    print(f"{'='*60}")
    
    # Process all files
    all_qa_pairs = []
    
    for txt_file in txt_files:
        qa_pairs = process_file(txt_file, config, num_pairs_per_chunk=args.num_pairs)
        all_qa_pairs.extend(qa_pairs)
    
    # Convert to desired format
    if args.format == 'alpaca':
        print("\n🔄 Converting to Alpaca format...")
        all_qa_pairs = convert_to_alpaca_format(all_qa_pairs)
    
    # Save output
    output_dir = Path(args.output_dir)
    output_file = output_dir / f"training_data_{args.format}.json"
    save_qa_pairs(all_qa_pairs, output_file)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ COMPLETE!")
    print(f"📊 Total QA pairs generated: {len(all_qa_pairs)}")
    print(f"📁 Output: {output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()