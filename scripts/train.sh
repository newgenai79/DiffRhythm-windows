export PYTHONPATH=$PYTHONPATH:$PWD

accelerate launch --config-file config/accelerate_config.yaml \
    train/train.py \
    --model-config config/diffrhythm-1b.json \
    --batch-size 6 \
    --max-frames 2048 \
    --min-frames 512 \
    --cond-drop-prob 0.2 \
    --style-drop-prob 0.2 \
    --lrc-drop-prob 0.2 \
    --audio-drop-prob 1.0 \
    --reset-lr 1 \
    --epochs 1000 \
    --resumable-with-seed 666 \
    --grad-accumulation-steps 1 \
    --grad-ckpt 0 \
    --exp-name diffrhythm-test \
    --file-path "dataset/train.scp" \