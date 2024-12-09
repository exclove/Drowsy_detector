### imports
import torch
import torch.nn as nn # functions s.t. have parameters. linear, convolution etc. 
import torch.optim as optim # optimizers
import torch.nn.functional as F  # functions s.t. don't have paramerters. relu, tanh etc.
from torch.utils.data import DataLoader # ex) minibatchs etc.
import torchvision.datasets as datasets # datasets pytorch presenting
import torchvision.transforms as transforms # transform
from data_customizing import eyeDataset
import torchvision.models as models
from global_name_space import get_args
import sys

### Set Device

if torch.cuda.is_available():
    device = torch.device("cuda") 
    print("CUDA is available")

elif torch.backends.mps.is_available():
    print("MPS is available")
    device = torch.device("mps") 

else :
    print("You can only use CPU")
    device = torch.device("cpu")


# Get arguments from globalnamespace.py
args = get_args()


### Hyperparameters
in_channel = args.in_channel
num_classes = args.num_classes
learning_rate = args.learning_rate
batch_size = args.batch_size
num_epochs = args.num_epochs
load_model = args.load_model
total_steps = args.num_epochs * (args.total_steps_num // args.batch_size)
gamma = args.gamma

### Define checkpoint
def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    # state: A dictionary containing the model’s state (state_dict) and the optimizer’s state.  
    # What is saved in state_dict:
	# •	For model.state_dict(): Model weights, biases, and other parameters.
	# •	For optimizer.state_dict(): The optimizer’s parameters (e.g., momentum terms, learning rate).


### Load Checkpoint
def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])



### Load data
my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


train_dataset = dataset = eyeDataset(csv_file = 'closed_open_labels.csv', 
                             root_dir = "eye_data",
                             transform = my_transforms)

train_set, test_set = torch.utils.data.random_split(dataset, [3500, 500])
train_loader = DataLoader(dataset = train_set, batch_size=batch_size, shuffle=True)

print(type(train_set))
print(type(train_loader))


test_loader = DataLoader(dataset = test_set, batch_size=batch_size, shuffle=True)



# Model 
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)



### Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Tadam(model.parameters(), total_steps=total_steps, gamma=gamma, lr=learning_rate)



if load_model :
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))

### Train Network
for epoch in range(num_epochs):
    losses = []

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    

    for batch_idx, (data, targets) in enumerate(train_loader):
        # data is tensor s.t. its shape is (# of batch_size, 1, 28, 28)
        data = data.to(device = device)
        targets = targets.to(device = device)

        # forward
        scores = model(data) # (64, 10)
        loss = criterion(scores, targets)
            # The loss function, criterion = nn.CrossEntropyLoss(), expects these raw scores (logits) as input.
	        # nn.CrossEntropyLoss internally applies softmax to convert the logits into probabilities and then calculates the loss based on the true labels.

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step

        # print(model.fc1.weight) # weights of fc1 before update

        optimizer.step()

        # print(model.fc1.weight) # weights of fc2 after update

 
    print(f"Loss at epoch {epoch} was {loss:.4f}")

    save_checkpoint(checkpoint, filename=f"checkpoint_epoch_{epoch}.pth.tar")

# the model’s parameters (e.g., weights of self.fc1 and self.fc2) are now updated.



### Check Accuracy on traing & test

def check_accuracy(loader, model):
    # if loader.dataset.train:
    #     print("Checking accuracy on training data")
    # else :
    #     print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader: 
            x = x.to(device)
            y = y.to(device)

            scores = model(x) 
            _, predictions = scores.max(1) # shape of "predictions" is (64, ) <-- Maximun value of each batch_size(number of pictures)
            num_correct += (predictions == y).sum() # shape of y is (batch_size, )
                # This counts how many predictions match the true labels by checking each 64 numbers
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
