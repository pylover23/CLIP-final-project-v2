from model import train_lora, evaluate_all, evaluate_prompts_text2image, ECommerceCLIPLinearProbe, load_best_linear_model

from dataset import ECommerceDataset, train_test_split
from torch.utils.data import DataLoader



if __name__ == "__main__":
    train_test_split(train_ratio=0.8)
    trainset = ECommerceDataset(csv_path="data_abo_subset_selected14_english/train.csv")
    testset = ECommerceDataset(csv_path="data_abo_subset_selected14_english/test.csv")
    batch_size = 128
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    # 训练LoRA模型，并保存权重
    lora_model = train_lora(train_loader, epochs=20, save_path="checkpoints/lora_model.pth")
    
    # 评估所有模型的性能，包括原始CLIP、线性探针和LoRA模型
    evaluate_all(test_loader, top_k=5)
    evaluate_all(test_loader, top_k=10)
    evaluate_all(test_loader, top_k=20)
    

    
