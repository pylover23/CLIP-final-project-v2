import os
import clip
import torch
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from model import OriginalCLIP, load_best_linear_model, load_lora_model
from dataset import ECommerceDataset
from torch.utils.data import DataLoader
from model import build_image_collection
import tqdm

app = FastAPI(title="CLIP 电商图文检索 Demo")

# ---------- 全局配置 ----------
IMAGE_DIR = "./data_abo_subset_selected14_english/images"
PERSIST_DIR = "./chroma_db_demo"
COLLECTION_NAME = "demo_collection"
TOP_K_DEFAULT = 5

# ---------- 加载模型 ----------
print("Loading model...")
MODEL = load_lora_model()  # 默认用 LoRA（效果最好），也可换 OriginalCLIP() 或 load_best_linear_model()
MODEL.eval()
print("Model loaded.")

# ---------- 构建向量库（image + text） ----------
from DBManager import VectorDBManager

dataset = ECommerceDataset(csv_path="data_abo_subset_selected14_english/abo_subset.csv")
data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

print("Building image collection...")
DB_IMAGE = build_image_collection(MODEL, data_loader, COLLECTION_NAME, PERSIST_DIR)
print(f"Image collection ready. Total: {DB_IMAGE.collection.count()}")

print("Building text collection...")
DB_TEXT = VectorDBManager(collection_name=COLLECTION_NAME + "_text", persist_dir=PERSIST_DIR)
if DB_TEXT.collection.count() == 0:
    tqdm_loader = tqdm.tqdm(data_loader, desc="Building text collection")
    for images, product_ids, prompts in tqdm_loader:
        images = images.to(MODEL.device)
        text_inputs = clip.tokenize(prompts["full"]).to(MODEL.device)
        with torch.no_grad():
            _, text_emb = MODEL.forward(None, text_inputs)
        DB_TEXT.add_image(
            [f"{int(pid)}_img" for pid in product_ids],
            text_emb
        )
    print(f"Text collection built. Total: {DB_TEXT.collection.count()}")
else:
    print(f"Text collection loaded from cache. Total: {DB_TEXT.collection.count()}")

# ---------- 加载商品信息用于展示 ----------
PRODUCT_DF = pd.read_csv("data_abo_subset_selected14_english/abo_subset.csv", usecols=["ProductId", "Brand", "ProductType", "Colour", "Material", "ProductTitle", "Image"])
PRODUCT_DF["ProductId"] = PRODUCT_DF["ProductId"].astype(int)
PRODUCT_MAP = {row["ProductId"]: row for _, row in PRODUCT_DF.iterrows()}


class SearchRequest(BaseModel):
    query: str
    top_k: int = TOP_K_DEFAULT
    hybrid: bool = False  # 是否启用 text+image 双路检索融合


class SearchItem(BaseModel):
    product_id: int
    title: str
    score: float
    image_url: str


@app.post("/api/search")
def search(req: SearchRequest):
    text_inputs = clip.tokenize([req.query]).to(MODEL.device)
    with torch.no_grad():
        _, text_emb = MODEL.forward(None, text_inputs)
    emb_list = text_emb.detach().cpu().numpy().tolist()

    if req.hybrid:
        # 双路检索：text→image + text→text，分数加权融合
        img_res = DB_IMAGE.collection.query(query_embeddings=emb_list, n_results=req.top_k * 3, where={"type": "image"})
        txt_res = DB_TEXT.collection.query(query_embeddings=emb_list, n_results=req.top_k * 3)

        img_scores = {pid: 1 - dist for pid, dist in zip(img_res["ids"][0], img_res["distances"][0])}
        txt_scores = {pid: 1 - dist for pid, dist in zip(txt_res["ids"][0], txt_res["distances"][0])}

        alpha = 0.6  # image 权重
        all_ids = set(img_scores) | set(txt_scores)
        fused = [(pid, alpha * img_scores.get(pid, 0) + (1 - alpha) * txt_scores.get(pid, 0)) for pid in all_ids]
        fused.sort(key=lambda x: x[1], reverse=True)
        top = fused[:req.top_k]
    else:
        # 单路检索：text→image
        results = DB_IMAGE.collection.query(query_embeddings=emb_list, n_results=req.top_k, where={"type": "image"})
        top = list(zip(results["ids"][0], [1 - d for d in results["distances"][0]]))

    items = []
    for pid_str, score in top:
        pid = int(pid_str.replace("_img", ""))
        info = PRODUCT_MAP.get(pid)
        title = info["ProductTitle"] if info is not None else str(pid)
        items.append(SearchItem(
            product_id=pid, title=title,
            score=round(score, 4),
            image_url=f"/images/{pid}.jpg"
        ))
    return {"query": req.query, "hybrid": req.hybrid, "results": items}


@app.get("/images/{product_id}.jpg")
def get_image(product_id: int):
    info = PRODUCT_MAP.get(product_id)
    image_name = info["Image"] if info is not None else f"{product_id}.jpg"
    path = os.path.join(IMAGE_DIR, image_name)
    if not os.path.exists(path):
        return {"error": "image not found"}
    return FileResponse(path, media_type="image/jpeg")


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>CLIP 电商图文检索</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "Segoe UI", sans-serif; background: #f5f5f5; color: #333; }
  .container { max-width: 960px; margin: 0 auto; padding: 24px; }
  h1 { text-align: center; margin-bottom: 24px; font-size: 1.6em; }
  .search-bar { display: flex; gap: 8px; margin-bottom: 24px; }
  .search-bar input {
    flex: 1; padding: 12px 16px; font-size: 16px;
    border: 1px solid #ccc; border-radius: 8px; outline: none;
  }
  .search-bar input:focus { border-color: #4a90d9; }
  .search-bar button {
    padding: 12px 24px; font-size: 16px; cursor: pointer;
    background: #4a90d9; color: #fff; border: none; border-radius: 8px;
  }
  .search-bar button:hover { background: #357abd; }
  .results { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 16px; }
  .card {
    background: #fff; border-radius: 8px; overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1); transition: transform 0.15s;
  }
  .card:hover { transform: translateY(-2px); }
  .card img { width: 100%; aspect-ratio: 1; object-fit: cover; background: #eee; }
  .card .info { padding: 8px 10px; font-size: 13px; }
  .card .title { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; line-height: 1.4; }
  .card .score { color: #888; font-size: 12px; margin-top: 4px; }
  .status { text-align: center; color: #999; margin: 40px 0; }
  .loading { display: none; text-align: center; margin: 20px 0; color: #999; }
</style>
</head>
<body>
<div class="container">
  <h1>CLIP 电商图文检索 Demo</h1>
  <div class="search-bar">
    <input id="query" type="text" placeholder="输入商品描述，如 a photo of a red dress" autofocus
           onkeydown="if(event.key==='Enter') doSearch()">
    <button onclick="doSearch()">搜索</button>
  </div>
  <div style="margin:-16px 0 16px; font-size:13px; color:#666;">
    <label><input type="checkbox" id="hybrid"> 双路检索（image + text 融合）</label>
  </div>
  <div class="loading" id="loading">检索中...</div>
  <div class="results" id="results"></div>
</div>
<script>
async function doSearch() {
  const q = document.getElementById('query').value.trim();
  if (!q) return;
  const box = document.getElementById('results');
  const loading = document.getElementById('loading');
  box.innerHTML = '';
  loading.style.display = 'block';
  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: q, top_k: 5, hybrid: document.getElementById('hybrid').checked})
    });
    const data = await res.json();
    loading.style.display = 'none';
    if (!data.results.length) {
      box.innerHTML = '<div class="status">未找到结果</div>';
      return;
    }
    box.innerHTML = data.results.map((r, i) => `
      <div class="card">
        <img src="${r.image_url}" alt="${r.title}" loading="lazy">
        <div class="info">
          <div class="title">${r.title}</div>
          <div class="score">#${i+1} &middot; ID: ${r.product_id} &middot; sim: ${r.score}</div>
        </div>
      </div>
    `).join('');
  } catch(e) {
    loading.style.display = 'none';
    box.innerHTML = '<div class="status">请求失败: ' + e.message + '</div>';
  }
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
