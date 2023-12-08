import argparse
import torch
torch.cuda.current_device()
import numpy as np
from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import MCGCL
from utils import *
import time

config_file = 'config.txt' 
with open(config_file) as i_f:
    i_f.readline()
    student_n, exercise_n, knowledge_n = i_f.readline().split(',')
    student_n, exercise_n, knowledge_n = int(student_n), int(exercise_n), int(knowledge_n)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='junyi', help='dataset name: junyi/Assent2015')
parser.add_argument('--res_file', default='./result/no_ss.txt', help='result')
parser.add_argument('--epoch', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--emb_size', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, help='l2 penalty')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--layers', type=float, default=2, help='the number of layer used')
parser.add_argument('--beta', type=float, default=1, help='ssl task maginitude')
parser.add_argument('--gama', type=float, default=1, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')
parser.add_argument('--shuffle', type=bool, default=True,help = 'shuffle datasets')
parser.add_argument('--exer_n', type=int, default=exercise_n, help='The number for exercise.')
parser.add_argument('--knowledge_n', type=int, default=knowledge_n, help='The number for knowledge concept.')
parser.add_argument('--student_n', type=int, default=student_n, help='The number for student.')
parser.add_argument('--gpu', type=int, default=0, help='The id of gpu, e.g. 0.')
parser.add_argument('--batchsize', type=int, default=256, help='batch')
args = parser.parse_args()
print(args)

device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')

def main():
    data_loader = TrainDataLoader(args)
    net = MCGCL(args)
    net = net.to(device)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    loss_function = net.loss_function
    optimizer = net.optimizer
    for epoch in range(args.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)

        start_time = time.time()

        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        pred_all, label_all = [], []
        correct_count, exer_count = 0, 0
        one_epoch_loss = 0.0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
            net.optimizer.zero_grad()
            output_1,losse, losss = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            loss = loss_function(torch.log(output+1e-10), labels)
            loss += losse + losss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0
            for i in range(len(labels)):
                if (labels[i] == 1 and output_1.view(-1)[i] > 0.5) or (labels[i] == 0 and output_1.view(-1)[i] < 0.5):
                    correct_count += 1
            pred_all += output_1.view(-1).to(torch.device('cpu')).tolist()
            label_all += labels.to(torch.device('cpu')).tolist()
            exer_count += len(labels)
            one_epoch_loss += loss.item()

        train_loss = one_epoch_loss / batch_count
        train_accuracy = correct_count / exer_count
        pred_all = np.array(pred_all)
        label_all = np.array(label_all)
        rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
        auc = roc_auc_score(label_all, pred_all)
        print('train_epoch: %d  train_loss : %f  train_acc: %f  train_auc: %f  train_rmse: %f' % (epoch, train_loss, train_accuracy, auc, rmse))
        
        with open(args.res_file, 'a', encoding='utf8') as f:
            f.write('------------------------------------------------------\n')
            f.write('epoch= %d \n' % epoch)
            f.write('layer= %d  batch=%d  lr=%f\n' % (args.layers, args.batchsize,args.lr))
            f.write('train_loss = %f , acc = %f , auc = %f , rmse = %f \n' % (loss, train_accuracy, auc,rmse))
        # test and save current model every epoch
        if epoch %10 == 0:
            save_snapshot(net, 'model/model_epoch' + str(epoch + 1))
        rmse, auc, acc = predict(args, net, epoch)
        print('test_epoch: %d  acc: %f  auc: %f  rmse: %f' % (epoch, acc, auc, rmse))
        
        with open(args.res_file, 'a', encoding='utf8') as f:
            f.write('------------------------------------------------------\n')
            f.write('epoch= %d \n' % epoch)
            f.write('layer= %d  batch=%d  lr=%f\n' % (args.layers, args.batchsize,args.lr))
            f.write('test_acc = %f , auc = %f , rmse = %f \n' % (acc, auc,rmse))

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

def predict(args, net, epoch):
   
    data_loader = ValTestDataLoader('predict')
    print('predicting model...')
    data_loader.reset()
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all, pred_label= [], [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        if batch_count == 1:
          output,_,_ = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs,1)
        else:
          output,_,_ = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
            pred_label.append(1) if output[i]>0.5 else pred_label.append(0)
                
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    print("saving scores and labels")
    torch.save(pred_all, 'dataset/pred_score.pt')
    torch.save(pred_label, 'dataset/pred_label.pt')
    torch.save(label_all, 'dataset/true_label.pt')

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
   
    accuracy = correct_count / exer_count
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))
    with open('result/ncd_model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch+1, accuracy, rmse, auc))

    return rmse, auc, accuracy


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


if __name__ == '__main__':
    main()
