import torch
from torch import nn
from tqdm.auto import tqdm


import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Train and validation
def one_step_train(model, train_dataloader, loss_fn, optimizer, device):
    model = model.to(device)

    model.train()

    train_loss, train_acc = 0, 0

    for i, batch in enumerate(train_dataloader):
        
        input_ids , attention_mask, targets = batch

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        # print(f'targets :\n {targets}')

        y_pred = model(
            input=input_ids,
            attention_mask=attention_mask
        )
        # print(f'y_pred :\n {y_pred}')

        loss = loss_fn(y_pred, targets)
        
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += ((y_pred_class == targets).sum().item())/len(y_pred)

    train_loss = train_loss/len(train_dataloader)
    train_acc = train_acc/len(train_dataloader)

    return train_loss, train_acc


def one_step_val(model, val_dataloader, loss_fn, device):

    model = model.to(device)

    model.eval()
    val_loss, val_acc = 0, 0

    with torch.inference_mode():

        for i, batch in enumerate(val_dataloader):
            
            input_ids , attention_mask, targets = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            y_pred = model(
                input=input_ids,
                attention_mask=attention_mask
            )
            
            loss = loss_fn(y_pred, targets)

            val_loss += loss.item()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            val_acc += ((y_pred_class == targets).sum().item())/len(y_pred)

        val_loss = val_loss/len(val_dataloader)
        val_acc = val_acc/len(val_dataloader)

    return val_loss, val_acc


def train(model,
                train_dataloader,
                val_dataloader,
                loss_fn_train,
                loss_fn_val,
                optimizer,
                device,
                epochs):

    results = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(epochs):

        train_loss, train_acc = one_step_train(model,
                                               train_dataloader,
                                               loss_fn_train, 
                                               optimizer,
                                               device)

        val_loss, val_acc = one_step_val(model,
                                         val_dataloader,
                                         loss_fn_val,
                                         device)

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

    return results



def evaluate_model(model, dataloader, device):
    model.eval()
    model.to(device)
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):

            input_ids , attention_mask, targets = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            # print(f'targets :\n {targets}')

            y_pred = model(
                input=input_ids,
                attention_mask=attention_mask
            )
            # _, predictions = torch.max(y_pred, 1)
            predictions = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    precision = precision_score(all_targets, all_predictions, average='macro', zero_division=1)
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    print(
        f"precision: {precision:.2f} | "
        f"recall: {recall:.2f} | "
        f"f1: {f1:.2f} \n "
    )

    cm = confusion_matrix(all_targets, all_predictions)

    class_labels = [i for i in range(cm.shape[0])]  # Assuming class labels are integers
    conf_matrix = confusion_matrix(all_targets, all_predictions, labels=class_labels)

    # Plot confusion matrix using seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.title("Confusion Matrix")
    plt.show()

    return precision, recall, f1, all_targets, all_predictions

