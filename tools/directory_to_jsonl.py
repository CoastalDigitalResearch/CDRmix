#!/usr/bin/env python3
"""
Convert directory structure to JSONL format for model training data preparation.
Modular architecture: File Discovery → Text Extraction → JSONL Writing
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import mimetypes

class TextExtractor:
    """Handles text extraction from different file types"""
    
    @staticmethod
    def extract_text(file_path):
        """Extract text content from file based on type"""
        try:
            # Handle JSON files (Common Pile format)
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Common Pile usually has 'text' field
                    return data.get('text', str(data))
            
            # Handle JSONL files
            elif file_path.suffix.lower() == '.jsonl':
                texts = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        texts.append(data.get('text', str(data)))
                return '\n'.join(texts)
            
            # Handle plain text files
            elif file_path.suffix.lower() in ['.txt', '.md', '.rst']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            # Handle other text-like files
            else:
                mime_type, _ = mimetypes.guess_type(str(file_path))
                if mime_type and mime_type.startswith('text/'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                
        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")
            return None
        
        return None

class FileDiscovery:
    """Handles recursive file discovery with filtering"""
    
    @staticmethod
    def find_files(root_dir, extensions=None):
        """Recursively find files with specified extensions"""
        if extensions is None:
            extensions = {'.json', '.jsonl', '.txt', '.md', '.rst'}
        
        root_path = Path(root_dir)
        files = []
        
        for file_path in root_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                files.append(file_path)
        
        return sorted(files)

class JSONLWriter:
    """Handles JSONL output with memory efficiency"""
    
    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def write_batch(self, texts, batch_size=1000):
        """Write texts to JSONL in batches"""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(tqdm(texts, desc="Writing JSONL")):
                if text and text.strip():  # Skip empty texts
                    json.dump({"text": text.strip()}, f, ensure_ascii=False)
                    f.write('\n')
                
                # Flush every batch_size items
                if (i + 1) % batch_size == 0:
                    f.flush()

def convert_directory_to_jsonl(input_dir, output_file, extensions=None, max_files=None):
    """
    Main conversion function - orchestrates the modular components
    
    Args:
        input_dir: Directory containing source files
        output_file: Output JSONL file path
        extensions: Set of file extensions to process
        max_files: Maximum number of files to process (for testing)
    """
    
    print(f"🔍 Discovering files in: {input_dir}")
    discoverer = FileDiscovery()
    files = discoverer.find_files(input_dir, extensions)
    
    if max_files:
        files = files[:max_files]
    
    print(f"📁 Found {len(files)} files to process")
    
    print(f"🔄 Extracting text content...")
    extractor = TextExtractor()
    texts = []
    
    for file_path in tqdm(files, desc="Processing files"):
        text = extractor.extract_text(file_path)
        if text:
            texts.append(text)
    
    print(f"📝 Extracted text from {len(texts)} files")
    
    print(f"💾 Writing to JSONL: {output_file}")
    writer = JSONLWriter(output_file)
    writer.write_batch(texts)
    
    print(f"✅ Conversion complete! {len(texts)} records written to {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert directory structure to JSONL for model training"
    )
    parser.add_argument("--input_dir", required=True, 
                       help="Directory containing source files")
    parser.add_argument("--output_file", required=True, 
                       help="Output JSONL file path")
    parser.add_argument("--extensions", nargs='+', 
                       default=['.json', '.jsonl', '.txt', '.md', '.rst'],
                       help="File extensions to process")
    parser.add_argument("--max_files", type=int, 
                       help="Maximum files to process (for testing)")
    
    args = parser.parse_args()
    
    # Convert extensions to set with leading dots
    extensions = {ext if ext.startswith('.') else f'.{ext}' for ext in args.extensions}
    
    convert_directory_to_jsonl(
        args.input_dir, 
        args.output_file, 
        extensions, 
        args.max_files
    )

