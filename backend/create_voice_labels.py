"""
Generate voice_labels.csv with both healthy and Parkinson's samples
"""
import os
import csv
from pathlib import Path

# Paths
healthy_dir = Path('datasets/voice_dataset/healthy')
parkinson_dir = Path('datasets/voice_dataset/parkinson')
output_csv = Path('datasets/voice_labels.csv')

# Collect all files
rows = []

# Add healthy samples (label = 0)
if healthy_dir.exists():
    for wav_file in healthy_dir.glob('*.wav'):
        rows.append({
            'filename': wav_file.name,
            'filepath': f'datasets/voice_dataset/healthy/{wav_file.name}',
            'parkinsons_label': 0,
            'category': 'healthy',
            'notes': 'Healthy voice sample'
        })

# Add Parkinson's samples (label = 1)
if parkinson_dir.exists():
    for wav_file in parkinson_dir.glob('*.wav'):
        rows.append({
            'filename': wav_file.name,
            'filepath': f'datasets/voice_dataset/parkinson/{wav_file.name}',
            'parkinsons_label': 1,
            'category': 'parkinsons',
            'notes': 'Parkinson-affected voice sample'
        })

# Write CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['filename', 'filepath', 'parkinsons_label', 'category', 'notes']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"✓ Generated {output_csv}")
print(f"  • Healthy samples: {sum(1 for r in rows if r['parkinsons_label'] == 0)}")
print(f"  • Parkinson's samples: {sum(1 for r in rows if r['parkinsons_label'] == 1)}")
print(f"  • Total: {len(rows)}")
