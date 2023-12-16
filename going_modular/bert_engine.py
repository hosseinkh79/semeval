import torch
from torch import nn
from tqdm.auto import tqdm


# Train and validation
def one_step_train(model, train_dataloader, loss_fn, optimizer, device):
    model = model.to(device)

    model.train()
    train_loss, train_acc = 0, 0

    for i, d in enumerate(train_dataloader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        y_pred = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        y_pred = y_pred.logits
        loss = loss_fn(y_pred, targets)
        
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if i % 50 == 0:
            print(f'i : {i}')

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

        for d in val_dataloader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            y_pred = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            y_pred = y_pred.logits
            loss = loss_fn(y_pred, targets)

            train_loss += loss.item()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            val_acc += ((y_pred_class == targets).sum().item())/len(y_pred)

        val_loss = val_loss/len(val_dataloader)
        val_acc = val_acc/len(val_dataloader)

    return val_loss, val_acc


def bert_train(model,
                train_dataloader,
                val_dataloader,
                loss_fn,
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
                                               loss_fn, optimizer,
                                               device)

        val_loss, val_acc = one_step_val(model,
                                         val_dataloader,
                                         loss_fn,
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
