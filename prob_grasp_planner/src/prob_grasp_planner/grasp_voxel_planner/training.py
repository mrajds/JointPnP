import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from grasp_success_network import GraspLKHNet, build_lkh
from grasp_data_loader import GraspDataLoader, convert_to_torch
import matplotlib.pyplot as plt

from tqdm import tqdm

pkg_path = '/home/mohanraj/ll4ma_prime_WS/src/prob_grasp_planner'
model_dir = '/models/reflex_grasp_inf_models/grasp_voxel_inf_net/'

class WeightedFocalLoss(torch.nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.1, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

#focal_loss = WeightedFocalLoss()

def train(model, device, data_loader, optimizer, epoch, weight):
    model.train()

    # freeze weights for encoder
    for name, param in model.named_parameters():
        if param.requires_grad and "voxel" in name:
            param.requires_grad = False

    avg_loss = 0
    num_batch = 0
    while data_loader.epochs_completed <= epoch:
        sample = convert_to_torch(data_loader.next_batch(256))
        optimizer.zero_grad()
        output = model(sample)
        #target = torch.zeros(output.shape, device=device)
        #sample['label'] = sample['label'].type(torch.int64)
        #target[(torch.arange(output.shape[0]), sample['label'].squeeze())] = 1
        target = sample['label'].type(torch.int64).squeeze()
        loss = F.nll_loss(output, target, weight)
        loss.backward()
        optimizer.step()
        avg_loss = avg_loss + loss.item()
        num_batch += 1
    avg_loss /= num_batch
    return avg_loss

def test(model, device, data_loader, weight=None):
    model.eval()
    test_loss = 0
    correct = 0
    if weight is None:
        weight = torch.tensor((1.0,1.0))
        weight = weight.to(device)

    with torch.no_grad():
        class_0_counter = 0
        class_1_counter = 0
        sample = convert_to_torch(data_loader.next_batch(data_loader.num_samples))
        target = sample['label'].type(torch.int64).squeeze()
        output = model(sample)
        test_loss += F.nll_loss(output, target, weight, reduction='sum').item()  # sum up batch loss
        #y = torch.zeros(output.shape[0], output.shape[1])
        #y[range(y.shape[0]), target] = 1
        #y = y.to(device)
        #test_loss += F.binary_cross_entropy_with_logits(output, y, weight)
        pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability
        #if pred.item() == 0:
        #    class_0_counter += 1
        #else:
        #    class_1_counter += 1

        correct += pred.eq(target.view_as(pred)).sum().item()
        TP = torch.sum((pred==0)&(target==0))
        FP = torch.sum((pred==0)&(target==1))
        FN = torch.sum((pred==1)&(target==0))

        precision = TP/(FP+TP)
        recall = TP/(FN+TP)
        F1 = 2*precision*recall/(precision + recall)

    test_loss /= data_loader.num_samples


    print(f'\nTest Set:\nPrecision={precision}\nRecall={recall}\nFScore={F1}')
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, data_loader.num_samples,
        100. * correct / data_loader.num_samples))
        
    return correct / data_loader.num_samples, test_loss, class_0_counter, class_1_counter

def main():
    torch.manual_seed(2021)
    device = torch.device("cuda")
    path = "/mnt/data_collection"

    #model = GraspLKHNet().to(device)
    model = build_lkh()

    train_data_path = '/home/mohanraj/reflex_grasp_data/processed/batch3/' + \
                        'grasp_voxelized_data.h5'

    test_data_path = '/home/mohanraj/reflex_grasp_data/processed/batch2/' + \
                        'grasp_voxelized_data.h5'
    
    grasp_loader = GraspDataLoader(train_data_path)
    test_loader = GraspDataLoader(test_data_path)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    weight = torch.tensor((1.0, 1.0))
    weight = weight.to(device)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[1200], 0.1)
    max_epoch = -1
    max_accuracy = -1
    nr_epochs = 300
    train_losses = np.zeros(nr_epochs)
    test_losses = np.zeros(nr_epochs)
    class_0_predictions = np.zeros(nr_epochs)
    class_1_predictions = np.zeros(nr_epochs)
    #accuracy, test_loss, class_0_counter, class_1_counter = test(model, device, test_loader)
    #exit(0)
    for epoch in tqdm(range(nr_epochs)):
        train_loss = train(model, device, grasp_loader, optimizer, epoch, weight)
        accuracy, test_loss, class_0_counter, class_1_counter = test(model, device, test_loader)
        accuracy_str = str(accuracy)
        if accuracy > max_accuracy:
            max_epoch = epoch
            max_accuracy = accuracy
            torch.save(model.state_dict(), pkg_path + model_dir + 'grasp_lkh')# + accuracy_str + "-" + str(epoch))

        if train_loss < 1e-5:
            break

        if train_loss > 1.0:
            train_loss = 1.0

        # used for logging
        train_losses[epoch] = train_loss
        test_losses[epoch] = test_loss
        class_0_predictions[epoch] = class_0_counter
        class_1_predictions[epoch] = class_1_counter

        if epoch % 5 == 0:
            print(train_loss, test_loss, class_0_counter, class_1_counter)
        
        if epoch % 100 == 0:
            fig, ax = plt.subplots()
            plt.xlabel("epoch")
            plt.ylabel("loss (NLL)")
            ax.plot(train_losses, 'g--', label='Training loss')
            ax.plot(test_losses, 'b--', label='Test loss')
            legend = ax.legend(loc='upper center')
            legend.get_frame().set_facecolor('C0')
            plt.savefig("losses(lr=0.01decay20).png", format="png")
            plt.close()

            fig, ax = plt.subplots()
            plt.xlabel("epoch")
            plt.ylabel("# predicted samples")
            ax.plot(class_0_predictions, 'r.', label='negative grasps')
            ax.plot(class_1_predictions, 'g.', label='positive grasps')
            legend = ax.legend(loc='upper center')
            legend.get_frame().set_facecolor('C0')
            plt.savefig("predicted-classes(lr=0.001decay20).png", format="png")
            plt.close()

        scheduler.step()
    print("max accuracy: " + str(max_accuracy))
    print("max epoch: " + str(max_epoch))

    torch.save(model.state_dict(), pkg_path + model_dir + 'grasp_lkh') 

    fig, ax = plt.subplots()
    plt.xlabel("epoch")
    plt.ylabel("loss (NLL)")
    ax.plot(train_losses, 'g--', label='Training loss')
    ax.plot(test_losses, 'b--', label='Test loss')
    legend = ax.legend(loc='upper center')
    legend.get_frame().set_facecolor('C0')
    plt.savefig("losses(lr=0.001decay20).png", format="png")
    #plt.show()
    plt.close()

    fig, ax = plt.subplots()
    plt.xlabel("epoch")
    plt.ylabel("# predicted samples")
    ax.plot(class_0_predictions, 'r.', label='negative grasps')
    ax.plot(class_1_predictions, 'g.', label='positive grasps')
    legend = ax.legend(loc='upper center')
    legend.get_frame().set_facecolor('C0')
    plt.savefig("predicted-classes(lr=0.001decay20).png", format="png")
    #plt.show()
    plt.close()



if __name__ == "__main__":
    main()
