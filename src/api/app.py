import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

from api.service import InferenceService, save_upload
from agent.nutrition_recommendation_agent import nutrition_recommendation_agent


# ---- 重要：避免你遇到的 OpenMP 冲突 ----
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

app = FastAPI(title="AutoVision Food Agent Demo")

# ====== 这里改成你自己的路径 ======
CKPT_PATH = "outputs/exp_food101_baseline/best.pt"
DATA_DIR = "data"
NUTRITION_JSON = "data/nutrition/food101_nutrition.json"

# 服务对象：启动时加载（只加载一次）
service = InferenceService(
    ckpt_path=CKPT_PATH,
    data_dir=DATA_DIR,
    nutrition_json_path=NUTRITION_JSON,
    model_name="convnext_tiny",
    num_classes=101,
    image_size=224,
    device="auto"
)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>AutoVision Food Agent</title>
      </head>
      <body style="font-family: Arial; max-width: 720px; margin: 40px auto;">
        <h2>AutoVision: Food Classification + Agent (CLIP) + Nutrition</h2>
        <form action="/predict" method="post" enctype="multipart/form-data">
          <input type="file" name="file" accept="image/*" required />
          <button type="submit">Upload & Predict</button>
        </form>
        <p style="color:#666;">Tip: first request may be slower (model/CLIP warmup).</p>
      </body>
    </html>
    """


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_path = save_upload(img_bytes, file.filename, out_dir="outputs_web")
    result = service.run(img_path, topk=5)

    return JSONResponse({
        "image_path": img_path,
        **result
    })

from typing import List


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    items = []

    # 1逐张图片：模型 + 视觉 Agent + 营养
    for f in files:
        img_bytes = await f.read()
        img_path = save_upload(img_bytes, f.filename, out_dir="outputs_web")

        result = service.run(img_path, topk=5)

        items.append({
            "image_path": img_path,
            "topk": result["topk"],
            "final_class": result["final_class"],
            "agent_reason": result["agent_reason"],
            "nutrition": result["nutrition"]
        })

    # 构造给“营养推荐 agent”的输入
    food_records = [
        {
            "food": it["final_class"],
            "nutrition": it["nutrition"]
        }
        for it in items
    ]

    #  调用营养推荐 agent
    recommendation = nutrition_recommendation_agent(food_records)

    #  返回：单图结果 + 总体推荐
    return JSONResponse({
        "items": items,
        "recommendation": recommendation
    })

