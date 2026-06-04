from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from PIL import Image
import random
import os

"""
    原数据集中商品类别字段部分是大写，如果直接作为文本属性输入模型，可能会导致模型无法正确理解
    因此需要进行一些预处理，将其转换为更自然的文本形式。
"""

PRODUCT_TYPE_TEXT = {
    "CELLULAR_PHONE_CASE": "cellular phone case",
    "CHAIR": "chair",
    "FINEEARRING": "fine jewelry earrings",
    "FINENECKLACEBRACELETANKLET": "fine jewelry necklace bracelet or anklet",
    "FINERING": "fine jewelry ring",
    "HOME": "home product",
    "HOME_BED_AND_BATH": "home bed and bath product",
    "HOME_FURNITURE_AND_DECOR": "home furniture and decor product",
    "KITCHEN": "kitchen product",
    "RUG": "rug",
    "SHOES": "shoes",
    "SOFA": "sofa",
    "TABLE": "table",
    "WALL_ART": "wall art",
}


def product_type_to_text(product_type):
    value = str(product_type or "").strip()
    if value in PRODUCT_TYPE_TEXT:
        return PRODUCT_TYPE_TEXT[value]
    return value.replace("_", " ").lower().strip() or "product"


class ECommerceDataset(Dataset):
    def __init__(self, transform=None, csv_path="./data_abo_subset_selected14_english/abo_subset.csv", max_samples=None,
                 image_dir="./data_abo_subset_selected14_english/images"):
        super().__init__()
        # 不需要在此处进行transform，可以直接使用CLIP的preprocess函数进行图像预处理
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_dir = image_dir
        self.df = pd.read_csv(
            csv_path,
            usecols=["ProductId", "ItemId", "Brand", "Category", "ProductType", "Colour", "Material", "ProductTitle", "Image"],
        )
        if max_samples is not None:
            self.df = self.df.head(max_samples).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        product_id = int(row["ProductId"])
        
        img = Image.open(os.path.join(self.image_dir, str(row["Image"]))).convert("RGB")
        # img = transforms.ToTensor()(img)  # 将PIL图像转换为Tensor，后续在训练循环中使用CLIP的preprocess进行进一步处理

        if self.transform:
            img = self.transform(img)
            
        product_type = product_type_to_text(row["ProductType"])
        brand = str(row["Brand"])
        material = str(row["Material"])
        product_title = str(row["ProductTitle"])
        
        # 根据不同的属性组合生成不同细粒度的prompt
        eme_set = {
            "type": f"a photo of a {product_type}",
            "material": f"a photo of a {material} {product_type}",
            "brand": f"a photo of a {brand} {product_type}",
            "brand_material": f"a photo of a {brand} {material} {product_type}",
            "full": f"a photo of a {product_title}",
        }
        
        return img, product_id, eme_set


def train_test_split(train_ratio=0.8, seed=42):
    """
        根据数据集中girls,boys,men,women四个子类别的ProductId进行划分
        以保证训练集和测试集中每个子类别的分布相似
    """
    # 如果已经存在则直接返回
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data_abo_subset_selected14_english")
    csv_path = os.path.join(data_dir, "abo_subset.csv")
    image_dir = os.path.join(data_dir, "images")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("train.csv and test.csv already exist, skipping train-test split.")
        return

    df = pd.read_csv(csv_path)
    # required_columns = ["ProductId", "ProductType", "Image"]
    # missing_columns = [col for col in required_columns if col not in df.columns]
    # if missing_columns:
    #     raise ValueError(f"Missing required columns: {missing_columns}")

    # 过滤掉图像文件不存在的样本，避免后续训练和评估时出现错误（下载图片时，原先的图片忘记删除了）
    df = df[df["Image"].apply(lambda name: os.path.exists(os.path.join(image_dir, str(name))))].copy()

    trainset_ids = []
    testset_ids = []
    # 数据集中总共有20组，按照ProductType进行分组，保证训练集和测试集中每个子类别的分布相似
    for _, group in df.groupby("ProductType", sort=True):
        product_ids = group["ProductId"].drop_duplicates().tolist()
        random.seed(seed)
        random.shuffle(product_ids)   # 打乱顺序，保证划分的随机性
        split_idx = int(len(product_ids) * train_ratio)
        trainset_ids.extend(product_ids[:split_idx])
        testset_ids.extend(product_ids[split_idx:])

    train_df = df[df["ProductId"].isin(trainset_ids)]
    test_df = df[df["ProductId"].isin(testset_ids)]
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved {len(train_df)} training samples to {train_path}")
    print(f"Saved {len(test_df)} test samples to {test_path}")


if __name__ == "__main__":
    train_test_split()