python src/predict.py --config configs/food101_baseline.yaml --ckpt outputs/exp_food101_baseline/best.pt --image demo\noodle.jpg

我实现了一个不确定性驱动的多模型协同 Agent，当主分类模型对相似食物不确定时，Agent 会主动调用 CLIP 进行视觉-文本验证，从而降低高置信错误。