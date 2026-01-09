python src/predict.py --config configs/food101_baseline.yaml --ckpt outputs/exp_food101_baseline/best.pt --image demo\noodle.jpg

我实现了一个不确定性驱动的多模型协同 Agent，当主分类模型对相似食物不确定时，Agent1 会主动调用 CLIP 进行视觉-文本验证，从而降低高置信错误。Agent2会根据上传食物的营养成分，给出搭配建议，支持多图片。

$env:PYTHONPATH="src"
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m uvicorn api.app:app --reload


python -m http.server 5500
http://127.0.0.1:5500/frontend.html
