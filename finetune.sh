python main.py \
  --data-root='{your-data-dir}' \
  --batch-size=32 \
  --n-splits=10 \
  --n-epochs=20 \
  --model-name='EEGNet' \
  --output-dir='results' \
  --ex-id="${1:-0}"