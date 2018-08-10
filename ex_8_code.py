#coral malachi - 314882853
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

#define parameters:
INPUT_SIZE_IMG = 784
FIRST_HIDDEN = 100
SECOND_HIDDEN = 50
OUTPUT_LAYER = 10
SIZE_BATCH = 40
NUM_OF_EPOCHS = 10
LRAENING_RATE = 0.01



###############################################################
#Function Name:print_message_each_epoch
#Function input:kind_of_set,length of set, loss of model, number
#of correct predictions of model and size of batch
#Function output:none
#Function Action:the function print a message to help the user
#follow the network progress
################################################################

def print_message_each_epoch(kind_of_set,set_len,m_loss,m_success,size_of_batch):
    print('\n' + kind_of_set + ': The average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        m_loss, m_success, (set_len * size_of_batch),
        100. * m_success / (set_len * size_of_batch)))

###############################################################
#Function Name: calculate_loss_print
#Function input: size_of_batch,model, set,is_training-boolean varible
#indicates id the set is trainning set or validation set
#Function output:loss
#Function Action:the function calculate the loss of the model
#for each epochs,and print write message
################################################################

def calculate_loss_print(size_of_batch,model,set,is_training):

    #boolean varible indicates id the set is trainning set or validation set
    print_kind_of_set="training set"
    if is_training != 1:
        print_kind_of_set = "validation set"
    #define varibles for loss, and number of correct prediction
    m_loss=0
    m_success=0
    #let the model know to switch to eval mode by calling .eval() on the model
    model.eval()
    for data,tag in set:
        #feed model with data
        model_result = model(data)
        #sum the loss
        m_loss = m_loss + F.nll_loss(model_result,tag,size_average=False).item()
        #call help function to get the right prediction
        y_tag = get_y_tag(model_result)
        #total of successfull predictions
        m_success += y_tag.eq(tag.data.view_as(y_tag)).cpu().sum()
    # save the len of training set in varible to save calls to len functions
    set_len = len(set)
    #calculate loss
    m_loss = m_loss/(size_of_batch*set_len)
    #call help function to print message about loss each epoch
    print_message_each_epoch(print_kind_of_set,set_len,m_loss,m_success,size_of_batch)
    return m_loss


###############################################################
#Function Name: get_y_tag
#Function input: model
#Function output:return the model prediction tag
#Function Action: the function return the prediction
#by getting  the index of the max log-probability
################################################################

def get_y_tag(model):
    return model.data.max(1, keepdim=True)[1]

def create_predictions_file(model, set):

    results_file = open("test2.pred", 'w')
    # let the model know to switch to eval mode by calling .eval() on the model
    model.eval()
    m_loss = 0
    #count number of success predictions
    num_of_success = 0
    for data, target in set:
        output = model(data)
        #sum the loss
        m_loss = m_loss + F.nll_loss(output, target, size_average=False).item()
        #call get_y_tag function to get prediction
        y_tag = get_y_tag(output)
        #update varible
        num_of_success = num_of_success + y_tag.eq(target.data.view_as(y_tag)).cpu().sum()
        #write to file current prediction according to the required format
        results_file.write(str(y_tag.item()) + "\n")
    #save the len of training set in varible to save calls to len functions
    set_len = len(set)
    #calaculate the final loss
    m_loss = m_loss / (set_len)

    print('\n Test_Set: the Average loss: {:.4f}, the Accuracy: {}/{} ({:.0f}%)\n'.format(
        m_loss, num_of_success, (set_len), 100. * num_of_success / (set_len)))
    #close prediction file
    results_file.close()



###############################################################
#Function Name:train_neural_network
#Function input:model, trainng and validation sets
#Function output:none
#Function Action:train on the training set and then test the
#network on the test set. This has the network make predictions on data it has never seen
################################################################

def train_neural_network(model,train,validation_set):

    #define 2 empty list to sve there the loss
    validation_set_scores = {}
    train_set_scores={}
    #set the optimizer
    optimizer=optim.Adagrad(model.parameters(),lr=LRAENING_RATE)
    for epoch in range(NUM_OF_EPOCHS):
        print "epoch number "+ str(epoch)
        model.train()

        for data, labels in train:
            optimizer.zero_grad()
            output = model(data)
            running_los = F.nll_loss(output,labels)
            running_los.backward()
            optimizer.step()
        #calculate the loss by calling calculate_loss_print function
        running_los = calculate_loss_print(SIZE_BATCH,model,train,1)
        train_set_scores[epoch+1]=running_los
        # calculate the loss by calling calculate_loss_print function
        running_los = calculate_loss_print(1,model,validation_set,0)
        validation_set_scores[epoch+1]=running_los
    first_lable, = plt.plot(validation_set_scores.keys(), validation_set_scores.values(), "g-", label='validation loss')
    second_lable, = plt.plot(train_set_scores.keys(), train_set_scores.values(), "r-", label='train loss')
    plt.legend(handler_map={first_lable: HandlerLine2D(numpoints=4)})
    plt.show()


#Model A: Neural Network with two hidden layers, the first layer should have a size of 100 and the
#second layer should have a size of 50, both should be followed by ReLU activation function
class NetA(nn.Module):

    def __init__(self,img_size_input):
        super(NetA,self).__init__()
        self.img_size_input=img_size_input
        self.fc0=nn.Linear(img_size_input,FIRST_HIDDEN)
        self.fc1=nn.Linear(FIRST_HIDDEN,SECOND_HIDDEN)
        self.fc2=nn.Linear(SECOND_HIDDEN,OUTPUT_LAYER)

    #the forward function
    def forward(self,x):
        x=x.view(-1,self.img_size_input)
        x=F.relu(self.fc0(x))
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return F.log_softmax(x,dim=1)

#Model B: Dropout - add dropout layers to model in A. You should place the dropout on the output of
#the hidden layers
class NetB(nn.Module):
    def __init__(self,img_size_input):
        super(NetB,self).__init__()
        self.img_size_input=img_size_input
        self.fc0=nn.Linear(img_size_input,FIRST_HIDDEN)
        self.fc1=nn.Linear(FIRST_HIDDEN,SECOND_HIDDEN)
        self.fc2=nn.Linear(SECOND_HIDDEN,OUTPUT_LAYER)

    #the forward function using dropout
    def forward(self,x):
        x=x.view(-1,self.img_size_input)
        x=F.relu(self.fc0(x))
        x=F.dropout(x,training=self.training)
        x=F.relu(self.fc1(x))
        x=F.dropout(x,training=self.training)
        x=self.fc2(x)
        return F.log_softmax(x,dim=1)

#Model C: Batch Normalization - add Batch Normalization layers to model in A. You should place the
#Batch Normalization before the activation functions
class NetC(nn.Module):
    def __init__(self,img_size_input):
        super(NetC,self).__init__()
        self.img_size_input=img_size_input
        self.fc0=nn.Linear(img_size_input,FIRST_HIDDEN)
        self.fc1=nn.Linear(FIRST_HIDDEN,SECOND_HIDDEN)
        self.fc2=nn.Linear(SECOND_HIDDEN,OUTPUT_LAYER)
        #Batch normalization reduces the amount by what the hidden unit values shift around
        self.first_batch_normalizition_layer = nn.BatchNorm1d(FIRST_HIDDEN)
        self.second_batch_normalizition_layer = nn.BatchNorm1d(SECOND_HIDDEN)

    # the forward function using Batch Normalization
    def forward(self,x):
        x=x.view(-1,self.img_size_input)
        x=self.first_batch_normalizition_layer(F.relu(self.fc0(x)))
        x=self.second_batch_normalizition_layer(F.relu(self.fc1(x)))
        x=self.fc2(x)
        return F.log_softmax(x,dim=1)


###############################################################
#Function Name:load_datasets_and_run_models
#Function input:none
#Function output:none
#Function Action:the function use torchvision inorder to uplode
#the MNIST training and test datasets. then,split the data (80:20)
#and build the network.
################################################################
def load_datasets_and_run_models():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    # Load the MNIST training and test datasets using torchvision
    training_set = datasets.FashionMNIST(root='./data',train=True, download=True,transform=transform)
    #save the len of training set in varible to save calls to len functions
    training_set_len = len(training_set)
    divided_size = int(0.2* training_set_len)

    set_test = datasets.FashionMNIST(train=False,root='./data',transform=transform)

    train_list = list(range(training_set_len))

    #random, non-contigous split
    validation_idx = np.random.choice(train_list,size=divided_size,replace=False)
    training_idx= list(set(train_list)-set(validation_idx))

    samples_train = SubsetRandomSampler(training_idx)
    samples_validation = SubsetRandomSampler(validation_idx)

    m_training_loader = torch.utils.data.DataLoader(batch_size=SIZE_BATCH,sampler=samples_train,dataset=training_set)
    m_validation_loader = torch.utils.data.DataLoader(batch_size=1,sampler=samples_validation,dataset=training_set)

    m_test_loader = torch.utils.data.DataLoader(dataset=set_test,shuffle=False,batch_size=1)
    #creat netC
    model = NetC(img_size_input=INPUT_SIZE_IMG)
    #call the main function of the program - training our network
    train_neural_network(model,m_training_loader,m_validation_loader)
    #call help function which create a predictions file
    create_predictions_file(model,m_test_loader)


if __name__ == "__main__":
    load_datasets_and_run_models()


