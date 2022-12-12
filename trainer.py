# can be added to the trainer later
# todo: add evaluation? maybe not
import torch
from torch.utils.tensorboard import SummaryWriter
import time


def train_normal(model, train_loader, criterion, optimizer, epoch_number, writer: SummaryWriter, PATH, device="cuda",
                  n_steps_show=100):
    train_iter_count = 0
    stime = time.time()
    for epoch in range(epoch_number):
        # running_loss = 0.0
        print(f"Epoch [{epoch}/{epoch_number}]\t")

        model.train()
        for step, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # outputs = outputs.cuda()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/Train", loss.item(), train_iter_count)
            train_iter_count += 1

            if (step + 1) % n_steps_show == 0:
                print(
                    f"Step [{step + 1}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")

            # print statistics
            # running_loss += loss.item()

        #             if i % 200 == 199:    # print every 2000 mini-batches
        #                 print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        #                 running_loss = 0.0
        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LR", lr, epoch)
    time_taken = (time.time() - stime) / 60
    print(f"Epoch [{epoch}/{epoch_number}]\t Time Taken: {time_taken} minutes")
    torch.save(model.state_dict(), PATH)

def test(testloader, model):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')