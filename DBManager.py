import chromadb
from chromadb.config import Settings
from torch import Tensor


class VectorDBManager:
    """
        文本和图像的embedding双向存储和检索管理器，使用ChromaDB作为底层向量数据库。
         - add_image: 添加图像embedding，metadata中type为"image"
         - add_text: 添加文本embedding，metadata中type为"text"
         - search_images: 根据文本embedding查询相似图像，返回对应的product_id列表
         - search_texts: 根据图像embedding查询相似文本，返回对应的product_id列表
         支持批量添加向量
    """
    
    def __init__(self, collection_name = "ecommerce_collection", persist_dir = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
    def add(self, product_ids, embeddings: list, metadatas):
        """ 通过 metadatas中的type区分文本和图像"""
        self.collection.add(
            ids=product_ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def add_image(self, product_ids, embeddings: Tensor):
        """ 支持单个传入图像的embedding，也支持批量传入图像的embedding """
        embeddings = embeddings.detach().cpu().numpy().tolist()
        self.collection.add(
            ids=product_ids,
            embeddings=embeddings,
            metadatas=[{"type": "image"} for _ in product_ids]
        )
    
    def add_text(self, product_ids, embeddings: Tensor):
        """ 支持单个传入文本的embedding，也支持批量传入文本的embedding """
        embeddings = embeddings.detach().cpu().numpy().tolist()
        self.collection.add(
            ids=product_ids,
            embeddings=embeddings,
            metadatas=[{"type": "text"} for _ in product_ids]
        )
    
    def search_images(self, query_embeddings: Tensor, top_k=5):
        """ 输入文本embedding，返回相似图像的product_id列表 """
        query_embeddings = query_embeddings.detach().cpu().numpy().tolist()
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            where={"type": "image"}
        )
        return results["ids"]
    
    def search_texts(self, query_embeddings: Tensor, top_k=5):
        """ 输入图像embedding，返回相似文本的product_id列表 """
        query_embeddings = query_embeddings.detach().cpu().numpy().tolist()
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            where={"type": "text"}
        )
        return results["ids"]