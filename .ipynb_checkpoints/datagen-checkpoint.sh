#!/bin/bash

# Multi-Domain Logical Reasoning Data Generation Pipeline
# Generates training data for: Syllogisms, Seating Arrangements, Family Tree Logic, Mixed Series

set -e  # Exit on error

echo "======================================"
echo "Multi-Domain Data Generation Pipeline"
echo "======================================"
echo ""

# Configuration
PROJECT_NAME="logical_reasoning_all_domains"
CONFIG_FILE="all_domains_config.yaml"

# Source URLs
URLS=(
  "https://www.csus.edu/faculty/d/dowden/_internal/_documents/logical-reasoning-12.pdf"
  "https://people.cs.umass.edu/~pthomas/solutions/Liar_Truth.pdf"
  "https://cdn.toprankers.net.in/docs/logical-reasoning-and-data-interpretation-for-cat-by-nishit-k-si-027ec9cbb8b72.pdf"
)

echo "📋 Step 1: Creating project directory structure..."
mkdir -p ${PROJECT_NAME}/{sources,data/{input,parsed,generated,curated,final}}
echo "✓ Directory structure created"
echo ""

echo "📥 Step 2: Downloading source materials..."
cd ${PROJECT_NAME}
for url in "${URLS[@]}"; do
  echo "  Downloading: $(basename $url)"
  wget -P sources/ -q --show-progress "$url"
done
echo "✓ All sources downloaded"
echo ""

echo "📋 Step 3: Copying sources to input directory..."
cp sources/* data/input/
echo "✓ Sources copied"
echo ""

echo "📖 Step 4: Ingesting and parsing documents..."
synthetic-data-kit ingest ./data/input/
echo "✓ Documents parsed"
echo ""

echo "🤖 Step 5: Generating Q&A pairs..."
echo "   (This will take ~10-15 minutes for 50 pairs per document)"
echo "   Tip: Add --verbose flag to see detailed progress"
echo ""
synthetic-data-kit -c ../${CONFIG_FILE} create ./data/parsed/ --type qa --num-pairs 50
echo "✓ Q&A pairs generated"
echo ""

echo "🔍 Step 6: Curating generated data..."
echo "   (Filtering for quality, threshold: 7.5/10)"
echo ""
synthetic-data-kit -c ../${CONFIG_FILE} curate ./data/generated/
echo "✓ Data curated"
echo ""

echo "💾 Step 7: Converting to training format..."
synthetic-data-kit -c ../${CONFIG_FILE} save-as ./data/curated/ --format alpaca
echo "✓ Data formatted"
echo ""

echo "📊 Step 8: Generating statistics..."
echo ""
echo "=== PIPELINE SUMMARY ==="
echo "Parsed documents: $(ls data/parsed/*.txt 2>/dev/null | wc -l)"
echo "Generated files: $(ls data/generated/*.json 2>/dev/null | wc -l)"
echo "Curated files: $(ls data/curated/*.json 2>/dev/null | wc -l)"
echo "Final training files: $(ls data/final/*.json 2>/dev/null | wc -l)"
echo ""

# Count total Q&A pairs
if [ -f data/final/*.json ]; then
  TOTAL_PAIRS=$(cat data/final/*.json | grep -o '"question"' | wc -l)
  echo "Total Q&A pairs: ${TOTAL_PAIRS}"
fi
echo ""

echo "✅ PIPELINE COMPLETE!"
echo ""
echo "📁 Training data location: ${PROJECT_NAME}/data/final/"
echo "🚀 Next step: Fine-tune your model using Unsloth"
echo ""
echo "To view sample questions:"
echo "  cat ${PROJECT_NAME}/data/final/*.json | head -50"