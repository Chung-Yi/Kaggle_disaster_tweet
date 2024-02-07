import torch

def get_predictions(model, dataloader, device, compute_acc=False):
    predictions = None
    correct = 0
    total = 0
      
    with torch.no_grad():

        for data in dataloader:
        
            if next(model.parameters()).is_cuda:
                data = [t.to(device) for t in data if t is not None]
            
    
            input_ids, attention_mask, token_type_ids, label_tensor = data

            
            outputs = model(input_ids=torch.squeeze(input_ids).to(device), 
                            token_type_ids=torch.squeeze(token_type_ids).to(device), 
                            attention_mask=torch.squeeze(attention_mask).to(device))
            
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)

            
            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[3].to(device)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
    
    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions

# if __name__ == '_main__':
    
#     # 讓模型跑在 GPU 上並取得訓練集的分類準確率
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("device:", device)
#     model = model.to(device)
#     _, acc = get_predictions(model, trainloader, compute_acc=True)
#     print("classification acc:", acc)