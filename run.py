import dataset
import models
import globals

import os
from tqdm import tqdm
import json
import torch
import torch.optim as optim
import torch.nn as nn
import time

training_params = json.load(open('config.json'))["training_params"]

# Creating Model and intializing
print("Creating model and intializing")
Nationality_model = models.Nationality_Model().float().to(globals.device)

idx2country = dataset.create_country_dict('idx')[1]

weights_list = [int(items.split('_')[2].rstrip('.pt')) for items in os.listdir(globals.WEIGHTS_DIR)]
weights_list = sorted(weights_list)
if weights_list == [] :
    models.initialize_embeddings(Nationality_model.embedding_layers, globals.device)
else :
    print("Loading weights for training epoch {}".format(weights_list[-1]))
    Nationality_model.load_state_dict(torch.load(os.path.join(globals.WEIGHTS_DIR, 'LSTM_Model1_{}.pt'.format(weights_list[-1]))))

if training_params["train_embeddings"] == "False" :
    print("Embeddings are not trainable")
    for items in Nationality_model.embedding_layers :
        items.weight.requires_grad = False


if training_params["is_train"] == "True" :
    print("----------------------------------------------------------------")
    print("Mode : Train")

    # Creating Dataset
    print("Creating Training Dataset")
    train_loader = dataset.create_dataloader('train', training_params["batch_size"], shuffle=True)
    valid_loader = dataset.create_dataloader('valid')

    # Training
    print("Training the model")
    optimizer = optim.Adam(Nationality_model.parameters(), lr = 0.01)
    criterion = nn.CrossEntropyLoss()

    train_logs = open(os.path.join(globals.LOG_DIR, 'training_logs.txt'), 'a')
    valid_logs = open(os.path.join(globals.LOG_DIR, 'validation_logs.txt'), 'a')

    max_accuracy = 0

    # lr initializing
    for param_groups in optimizer.param_groups:
        param_groups['lr'] = param_groups['lr'] * (training_params['decay_rate']**(training_params["start_epoch"] - 1))

    print('------------------------------------------------------------------------------')

    for epoch in range(training_params["start_epoch"], training_params["end_epoch"] + 1) :

        # if epoch % training_epoch["decay_epoch_size"] == 0:
        for param_groups in optimizer.param_groups:
            param_groups['lr'] = param_groups['lr'] * training_params['decay_rate']
        print("Updated learning rate : ", optimizer.param_groups[0]['lr'])
        start_time = time.time()
        # TRAINING
        total_loss = 0
        Nationality_model.train()
        for idx, (data_sample) in enumerate(train_loader):
            Nationality_model.zero_grad()
            data_sample = [items.to(globals.device).long() for items in data_sample]
            output = Nationality_model(data_sample[:3])
            loss = criterion(output, data_sample[3])
            loss.backward()
            nn.utils.clip_grad_norm_(Nationality_model.parameters(), 5)
            optimizer.step()
            total_loss += loss.item()
            # print('Epoch : {}/{}, Iteration : {}/{}, Loss : {}' \
            #     .format(epoch, training_params["end_epoch"], idx + 1, len(train_loader),loss.item()))
        string = 'Training Epoch : {}/{}, Epoch Loss : {}' \
                .format(epoch, training_params["end_epoch"], total_loss/len(train_loader)) 
        print(string)
        train_logs.write(string+"\n")
        train_logs.flush()
        if epoch % 2 == 0 :
            torch.save(Nationality_model.state_dict(), os.path.join(globals.WEIGHTS_DIR, "LSTM_Model1_{}.pt".format(epoch)))

        ## VALIDATION
        Nationality_model.eval()
        with torch.no_grad() :
            accuracy = 0
            correct_list = []
            for idx, (data_sample) in enumerate(valid_loader):
                data_sample = [items.to(globals.device).long() for items in data_sample]
                output = Nationality_model(data_sample[:3])
                _, top1pred = torch.max(output, 1)
                if top1pred.item() == data_sample[3].item():
                    accuracy += 1
                correct_list.append((idx, top1pred.item(), data_sample[3].item()))
            if accuracy > max_accuracy :
                max_accuracy = accuracy
                best_metric_logs = open(os.path.join(globals.LOG_DIR, 'best_metric_results.txt'), 'w')
                best_metric_logs.write("EPOCH: {}/{}, Accuracy : {}\n"
                                        .format(epoch, training_params["end_epoch"], accuracy*100/len(valid_loader)))
                best_metric_logs.write('------------------------------------------------------------------\n\n')
                for items in correct_list :
                    best_metric_logs.write('Index : {}, Predicted : {}, Actual : {}\n'
                                        .format(items[0], idx2country[items[1]], idx2country[items[2]]))
                best_metric_logs.flush()
                best_metric_logs.close()
            string = 'Validating Epoch : {}/{}, Accuracy : {}'\
                        .format(epoch, training_params["end_epoch"], accuracy*100/len(valid_loader))
            print(string)
            valid_logs.write(string+'\n')
            valid_logs.flush()
        print("Time for completion of epoch : {} seconds".format((time.time()-start_time)))
        print("------------------------------------------------------------------")
    train_logs.close()
    valid_logs.close()

else :
    print("----------------------------------------------------------------")
    print("Mode : Test")

    # Creating Dataset
    print("Creating Testing Dataset")
    test_loader = dataset.create_dataloader('test', shuffle=False)

    test_logs = open(os.path.join(globals.LOG_DIR, 'testing_logs.txt'), 'w')

    Nationality_model.eval()
    print("Testing the model")
    with torch.no_grad() :
        accuracy = 0
        correct_list = []
        for idx, (data_sample) in enumerate(test_loader):
            data_sample = [items.to(globals.device).long() for items in data_sample]
            output = Nationality_model(data_sample[:3])
            _, top1pred = torch.max(output, 1)
            if top1pred.item() == data_sample[3].item():
                accuracy += 1
            correct_list.append((idx, top1pred.item(), data_sample[3].item()))
        string = 'Testing Accuracy after training for {} epochs : {}' \
                  .format(weights_list[-1], accuracy*100/len(test_loader))
        print(string)
        test_logs.write(string+'\n')
        test_logs.write('------------------------------------------------------------------\n\n')
        for items in correct_list :
            test_logs.write('Index : {}, Predicted : {}, Actual : {}\n'
                            .format(items[0], idx2country[items[1]], idx2country[items[2]]))
        test_logs.write('------------------------------------------------------------------\n\n')
        test_logs.flush()
        test_logs.close()






    