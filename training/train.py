import torch


def train_model(model,
                train_loader,
                criterion,
                optimizer,
                device,
                epochs=3,
                ewc=None,
                lambda_ewc=0):

    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            # Add EWC loss if available
            if ewc is not None:
                loss += lambda_ewc * ewc.penalty(model)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")