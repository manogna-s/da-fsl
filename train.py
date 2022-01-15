import os
import argparse
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from data_utils.data_provider import load_images
from models.res18_film import resnet18_film
from models.res18 import resnet18
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from easyfsl.data_tools import TaskSampler
from easyfsl.utils import plot_images, sliding_average

#==============eval
def evaluate(model, input_loader):
    model.eval()
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    for i in range(num_iter):
        inputs, labels = iter_test.next()
        inputs, labels = inputs.cuda(), labels.cuda()

        probabilities = model(inputs)

        probabilities = probabilities.data.float()
        labels = labels.data.float()
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)

    print(all_probs.shape, all_labels.shape)
    _, predict = torch.max(all_probs, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels) / float(all_labels.size()[0])

    return accuracy.data.item() 

def base_training(args):
    base_ckpt_path = os.path.join(args.log, 'base_ckpt') 
    os.makedirs(base_ckpt_path, exist_ok=True)

    train_dataset, train_loader = load_images(args.src_file, resize_size=100, is_train=True, crop_size=84, batch_size=32, is_cen=True)

    ckpt_path = 'data/pretrained_ckpt/imagenet-net/model_best.pth.tar'
    model = resnet18_film(pretrained=True, pretrained_model_path=ckpt_path, classifier='linear', num_classes=30).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9) 
    lr_scheduler = CosineAnnealingLR(optimizer, 10)

    n_epochs = 50
    best_acc = 0
    for ep in range(n_epochs):
        model.train()
        iter_dataloader = iter(train_loader)
        for inputs, labels in iter_dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            logits = model(inputs)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        acc = evaluate(model, train_loader)
        print(f'Epoch: {ep}, Train acc:{acc}')
        if acc>best_acc:
            best_acc = acc
            torch.save({'state_dict': model.state_dict(),
                        'base_classifier': model.cls_fn.state_dict(),},
                        os.path.join(base_ckpt_path, f"best_ckpt.pth"))
    
    return

def fsl_training(args):
    test_dataset, test_target_loader = load_images(args.tgt_file, resize_size=100, is_train=False, crop_size=84, batch_size=32)
    base_ckpt_path = os.path.join(args.log, 'base_ckpt', 'best_ckpt.pth')

    N_WAY = 5 # Number of classes in a task
    N_SHOT = 5 # Number of images per class in the support set
    N_QUERY = 10 # Number of images per class in the query set
    N_EVALUATION_TASKS = 100

    # The sampler needs a dataset with a "labels" field. Check the code if you have any doubt!
    test_sampler = TaskSampler(test_dataset, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS)

    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=12,
                            pin_memory=True, collate_fn=test_sampler.episodic_collate_fn,)

    protos_acc = []
    tasks_acc = []
    for  i in range(N_EVALUATION_TASKS):
        (support_images, support_labels, query_images, query_labels, class_ids,) = next(iter(test_loader))
        support_images, support_labels, query_images, query_labels = support_images.cuda(), support_labels.cuda(), query_images.cuda(), query_labels.cuda()

        n_way = len(torch.unique(support_labels))
        model = resnet18_film(pretrained=True, pretrained_model_path=base_ckpt_path, classifier='linear', num_classes=n_way).cuda()
        film_params = model.get_parameters()
        optimizer = torch.optim.SGD(film_params, lr=0.1, momentum=0.9)

        support_features = model.embed(support_images)

        proto = torch.cat([support_features[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)])
        model.cls_fn.weight = nn.Parameter(proto)
        query_features = model.embed(query_images)
        query_logits = model.cls_fn(query_features)
        _, query_preds = torch.max(F.softmax(query_logits, dim=1), dim=1)
        protos_task_accuracy = torch.sum(torch.squeeze(query_preds).float() == query_labels) / float(query_labels.size()[0])
        protos_task_accuracy = protos_task_accuracy.data.item()
        protos_acc.append(protos_task_accuracy)

        for  k in range(5):
            optimizer.zero_grad()
            support_logits = model(support_images)
            loss = nn.CrossEntropyLoss()(support_logits, support_labels)
            loss.backward()
            optimizer.step()

        query_features = model.embed(query_images)
        query_logits = model.cls_fn(query_features)
        _, query_preds = torch.max(F.softmax(query_logits, dim=1), dim=1)
        task_accuracy = torch.sum(torch.squeeze(query_preds).float() == query_labels) / float(query_labels.size()[0])
        task_accuracy = task_accuracy.data.item()
        tasks_acc.append(task_accuracy)
        # print(support_images.shape, support_labels.shape, query_images.shape, query_labels, class_ids)

    protos_acc = np.array(protos_acc) * 100
    protos_mean_acc = protos_acc.mean()
    protos_conf = (1.96 * protos_acc.std()) / np.sqrt(len(protos_acc))
    print(f'Using prototypes: Average acc over {N_EVALUATION_TASKS} tasks: {protos_mean_acc}+/- {protos_conf} %')

    acc = np.array(tasks_acc) * 100
    mean_acc = acc.mean()
    conf = (1.96 * acc.std()) / np.sqrt(len(acc))
    print(f'Updating FiLM params: Average acc over {N_EVALUATION_TASKS} tasks: {mean_acc}+/- {conf} %')

    # plot_images(support_images, "support images", images_per_row=N_SHOT)
    # plot_images(query_images, "query images", images_per_row=N_QUERY)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', type=str, help='all sets of configuration parameters',
                        default='/home/manogna/da-fsl/exps/v1/data_list/Real_World_1-30.txt')
    parser.add_argument('--tgt_file', type=str, help='all sets of configuration parameters',
                        default='/home/manogna/da-fsl/exps/v1/data_list/tgt_Clipart.txt')
    parser.add_argument('--log', type=str, help='all sets of configuration parameters',
                        default='/home/manogna/da-fsl/exps/v1')
    parser.add_argument('--base_train', type=bool, help='all sets of configuration parameters',
                        default=False) 
    parser.add_argument('--fsl_train', type=bool, help='all sets of configuration parameters',
                        default=False)                    
    args = parser.parse_args()

    if args.base_train:
        base_training(args)
    if args.fsl_train:
        fsl_training(args)
