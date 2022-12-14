# can be added to the trainer later
# todo: add evaluation? maybe not
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from utils import *
#
#
# def train_normal(model, train_loader, criterion, optimizer, epoch_number, writer: SummaryWriter,
#                  PATH, device="cuda",
#                   n_steps_show=100):
#     train_iter_count = 0
#     stime = time.time()
#     for epoch in range(epoch_number):
#         # running_loss = 0.0
#         print(f"Epoch [{epoch}/{epoch_number}]\t")
#
#         model.train()
#         for step, data in enumerate(train_loader, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data
#             inputs = inputs.cuda()
#             labels = labels.cuda()
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # forward + backward + optimize
#             outputs = model(inputs)
#             # outputs = outputs.cuda()
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             writer.add_scalar("Loss/Train", loss.item(), train_iter_count)
#             train_iter_count += 1
#
#             if (step + 1) % n_steps_show == 0:
#                 print(
#                     f"Step [{step + 1}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")
#
#             # print statistics
#             # running_loss += loss.item()
#
#         #             if i % 200 == 199:    # print every 2000 mini-batches
#         #                 print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#         #                 running_loss = 0.0
#         lr = optimizer.param_groups[0]["lr"]
#         writer.add_scalar("LR", lr, epoch)
#     time_taken = (time.time() - stime) / 60
#     print(f"Epoch [{epoch}/{epoch_number}]\t Time Taken: {time_taken} minutes")
#     torch.save(model.state_dict(), PATH)
#
# # +
# # def train_adv(model, img_tensor_list, label_tensor_list, criterion, optimizer, epoch_number, writer: SummaryWriter,
# #                  PATH, device="cuda",
# #                   n_steps_show=100, batch_size = 100):
# #     train_iter_count = 0
# #     stime = time.time()
# #     for epoch in range(epoch_number):
# #         # running_loss = 0.0
# #         print(f"Epoch [{epoch}/{epoch_number}]\t")
#
# #         model.train()
# #         for step in range(0, len(img_tensor_list), batch_size):
# #             # get the inputs; data is a list of [inputs, labels]
# #             inputs = img_tensor_list[step :step + batch_size]
# #             labels = label_tensor_list[step : step + batch_size]
# #             inputs = inputs.cuda()
# #             labels = labels.cuda()
# #             # zero the parameter gradients
# #             optimizer.zero_grad()
#
# #             # forward + backward + optimize
# #             outputs = model(inputs)
# #             # outputs = outputs.cuda()
# #             loss = criterion(outputs, labels)
# #             loss.backward()
# #             optimizer.step()
#
# #             writer.add_scalar("Loss/Train", loss.item(), train_iter_count)
# #             train_iter_count += 1
#
# #             if (step) % n_steps_show == 0:
# #                 print(
# #                     f"Step [{step + 1}/{len(img_tensor_list)}]\t Loss: {round(loss.item(), 5)}")
#
# #             # print statistics
# #             # running_loss += loss.item()
#
# #         #             if i % 200 == 199:    # print every 2000 mini-batches
# #         #                 print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
# #         #                 running_loss = 0.0
# #         lr = optimizer.param_groups[0]["lr"]
# #         writer.add_scalar("LR", lr, epoch)
# #     time_taken = (time.time() - stime) / 60
# #     print(f"Epoch [{epoch}/{epoch_number}]\t Time Taken: {time_taken} minutes")
# #     torch.save(model.state_dict(), PATH)
# # -
#
# def train_adv(model, train_loader, train_criterion, val_vriterion,
#               optimizer, epoch_number, attack, writer: SummaryWriter,
#                  PATH, device="cuda",
#                   n_steps_show=100):
#
#     train_iter_count, val_iter_count = 0
#     stime = time.time()
#     for epoch in range(epoch_number):
#         # running_loss = 0.0
#         print(f"Epoch [{epoch}/{epoch_number}]\t")
#
#         model.train()
#
#         tr_loss_epoch = []
#
#         for step, data in enumerate(train_loader, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data
#             inputs = inputs.cuda()
#             labels = labels.cuda()
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # forward + backward + optimize
#
#
#             adv_inputs = generate_adv(model, attack, inputs, labels)
#             outputs = model(adv_inputs)
#             # outputs = outputs.cuda()
#             loss = train_criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             tr_loss_epoch.append(loss.item())
#
#             writer.add_scalar("Loss/Train", loss.item(), train_iter_count)
#             train_iter_count += 1
#
#             if (step + 1) % n_steps_show == 0:
#                 print(
#                     f"Step [{step + 1}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")
#
#             # print statistics
#             # running_loss += loss.item()
#
#         #             if i % 200 == 199:    # print every 2000 mini-batches
#         #                 print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#         #                 running_loss = 0.0
#         lr = optimizer.param_groups[0]["lr"]
#         writer.add_scalar("LR", lr, epoch)
#
#         model.eval()
#         with torch.no_grad():
#             val_loss_epoch = []
#             for step, (inputs_val, labels_val) in enumerate(val_loader):
#
#                 inputs_val = inputs_val.to(device)
#                 labels_val = labels_val.to(device)
#
#                 stand_outputs = model(inputs_val) #todo: is this correct?
#
#
#                 loss = val_criterion(stand_outputs, labels_val)
#
#                 # Logging & show validation statistics
#                 val_loss_epoch.append(loss.item())
#                 writer.add_scalar("Loss/Eval", loss.item(), val_iter_count)
#                 val_iter_count += 1
#
#         # Checkpointing
# #         if (epoch+1) % n_epoch_checkpoint == 0:
# #             checkpoint(model, [train_batch_size, test_batch_size], optimizer, mainscheduler, current_epoch,
# #                        logger, "SimCLR_epoch_{}_checkpoint.pt")
#
#         # Logging & Show epoch-level statistics
#
#         writer.add_scalars("Loss (Epoch)", {
#             'Train': np.mean(tr_loss_epoch),
#             'Eval': np.mean(val_loss_epoch)
#         }, epoch)
#         print(
#             f"Epoch [{epoch+1}/{epoch_number}]\t Training Loss: {np.mean(tr_loss_epoch)}\t lr: {round(lr, 5)}")
#         print(
#             f"Epoch [{epoch+1}/{epoch_number}]\t Validation Loss: {np.mean(val_loss_epoch)}\t lr: {round(lr, 5)}")
#         current_epoch += 1
#
#         time_taken = (time.time() - stime) / 60
#     print(f"Epoch [{epoch}/{epoch_number}]\t Time Taken: {time_taken} minutes")
#     torch.save(model.state_dict(), PATH)
#

def train(model, train_loader, train_criterion,val_loader, val_criterion,
              optimizer, epoch_number, writer: SummaryWriter, 
                 PATH, adv_train, attack, device="cuda",
                  n_steps_show=100):
    
    train_iter_count, val_iter_count = 0,0
    current_epoch = 0
    stime = time.time()
    for epoch in range(epoch_number):
        # running_loss = 0.0
        print(f"Epoch [{epoch}/{epoch_number}]\t")

        model.train()
        
        tr_loss_epoch = []
        
        for step, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            
            if adv_train:
                adv_inputs = attack(inputs, labels)
                adv_inputs = adv_inputs.to(device)
                outputs = model(adv_inputs)
            else:
                
                outputs = model(inputs)
            # outputs = outputs.cuda()
            loss = train_criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            tr_loss_epoch.append(loss.item())

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

        model.eval()
        with torch.no_grad():
            val_loss_epoch = []
            for step, (inputs_val, labels_val) in enumerate(val_loader):
                
                inputs_val = inputs_val.to(device)
                labels_val = labels_val.to(device)
                
                stand_outputs = model(inputs_val) #todo: is this correct?  
                
        
                loss = val_criterion(stand_outputs, labels_val)

                # Logging & show validation statistics
                val_loss_epoch.append(loss.item())
                writer.add_scalar("Loss/Eval", loss.item(), val_iter_count)
                val_iter_count += 1

        # Checkpointing
#         if (epoch+1) % n_epoch_checkpoint == 0:
#             checkpoint(model, [train_batch_size, test_batch_size], optimizer, mainscheduler, current_epoch,
#                        logger, "SimCLR_epoch_{}_checkpoint.pt")

        # Logging & Show epoch-level statistics

        writer.add_scalars("Loss (Epoch)", {
            'Train': np.mean(tr_loss_epoch),
            'Eval': np.mean(val_loss_epoch)
        }, epoch)
        print(
            f"Epoch [{epoch+1}/{epoch_number}]\t Training Loss: {np.mean(tr_loss_epoch)}\t lr: {round(lr, 5)}")
        print(
            f"Epoch [{epoch+1}/{epoch_number}]\t Validation Loss: {np.mean(val_loss_epoch)}\t lr: {round(lr, 5)}")
        current_epoch += 1

        time_taken = (time.time() - stime) / 60
    print(f"Epoch [{epoch}/{epoch_number}]\t Time Taken: {time_taken} minutes")
    torch.save(model.state_dict(), PATH)
    pt_path = "."+PATH.split(".")[-2] + ".pt"
    torch.save(model, pt_path) #PT VERSION MODEL



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


