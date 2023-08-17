import os.path
import fire
from torch import nn
from torch.autograd import Variable
from data import dataset, dataloader
from transformers import get_cosine_schedule_with_warmup, AdamW
from models import ReadabilityModel
from config import DefaultConfig
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
import warnings
import torch

warnings.filterwarnings("ignore")
config = DefaultConfig()


def setup_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def train(**kwards):
    config.parse(kwards)
    annealing_factor = config.annealing_factor
    folder = './model_save'
    if not os.path.exists(folder):
        os.makedirs(folder)
    log_name = config.dataset_name
    trainingLog_handle = open(folder + '/training_log_' + log_name + '_factor_' + str(annealing_factor) + '.txt',
                              mode='w')
    trainingLog_handle.write('\n')
    trainingLog_handle.write("config: \n")
    trainingLog_handle.write("{ \n")
    for key, value in config.__class__.__dict__.items():
        if not key.startswith('__'):
            if key != 'bert_hidden_size':
                trainingLog_handle.write('     ' + key + ' = ' + str(getattr(config, key)) + '\n')
    trainingLog_handle.write("} \n")
    trainingLog_handle.write('\n')

    setup_seed(config.seed)

    train_dataset = dataset.TrainDataset(config.root_dir_train, config=config, feature_file=config.feature_train)
    train_loader = dataloader.MyDataloader(dataset=train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                           num_workers=16, config=config)
    test_dataset = dataset.TrainDataset(config.root_dir_test, config=config, feature_file=config.feature_test)
    test_loader = dataloader.MyDataloader(dataset=test_dataset, batch_size=config.test_batch_size, shuffle=True,
                                          num_workers=16, config=config)

    readabilityModel = ReadabilityModel()
    readabilityModel = readabilityModel.cuda()
    print(readabilityModel)
    trainingLog_handle.write(str(readabilityModel))
    trainingLog_handle.write('\n')

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(readabilityModel.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.schedule:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                    num_training_steps=config.max_epoch)

    best_test = 0
    for epoch in range(1, config.max_epoch + 1):

        running_loss = 0.0
        for i, train_data in enumerate(train_loader, 1):
            input_ids, attention_mask, input_ids_sentence, attention_mask_sentence, feature, label = train_data
            input_ids = Variable(input_ids)
            attention_mask = Variable(attention_mask)
            input_ids_sentence = Variable(input_ids_sentence)
            attention_mask_sentence = Variable(attention_mask_sentence)
            feature = Variable(feature)
            label = Variable(torch.tensor([*label]))
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            feature = feature.cuda()
            label = label.cuda()
            input_ids_sentence = input_ids_sentence.cuda()
            attention_mask_sentence = attention_mask_sentence.cuda()
            optimizer.zero_grad()

            class_out, final_represention, cls_embedding, document_de, hidden_sen_mix, sentence_de, hidden_w_mix, word_de, feature, feature_norm, feature_de, share_represention_de = readabilityModel(
                (input_ids, attention_mask, input_ids_sentence, attention_mask_sentence, feature))
            # Restructuring losses of public information
            common_KLD = nn.functional.mse_loss(share_represention_de, cls_embedding.detach(),
                                                reduction='mean') + nn.functional.mse_loss(share_represention_de,
                                                                                           hidden_sen_mix.detach(),
                                                                                           reduction='mean') + nn.functional.mse_loss(
                share_represention_de, hidden_w_mix.detach(), reduction='mean') + nn.functional.mse_loss(
                share_represention_de, feature.detach(),
                reduction='mean')
            # Reconstruction loss of documents based on private information
            du_KLD = nn.functional.mse_loss(document_de, cls_embedding.detach(), reduction='mean')
            # Reconstruction loss of sentences based on private information
            sen_KLD = nn.functional.mse_loss(sentence_de, hidden_sen_mix.detach(), reduction='mean')
            # Reconstruction loss of words based on private information
            word_KLD = nn.functional.mse_loss(word_de, hidden_w_mix.detach(), reduction='mean')
            # Reconstruction loss of linguistic features based on private information
            feature_KLD = nn.functional.mse_loss(feature_de, feature_norm.detach(), reduction='mean')
            # Restructuring loss of private information
            private_KLD = du_KLD + sen_KLD + word_KLD + feature_KLD
            loss = criterion(class_out,
                             label) + annealing_factor * common_KLD + annealing_factor * private_KLD  # united loss
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            if i % (500 // config.train_batch_size) == 0:
                print('[%d, %5d] loss: %.3f' % (
                    epoch, i * config.train_batch_size, running_loss / (500 // config.train_batch_size)))
                trainingLog_handle.write('[%d, %5d] loss: %.3f' % (
                    epoch, i * config.train_batch_size, running_loss / (500 // config.train_batch_size)))
                trainingLog_handle.write('\n')
                running_loss = 0.0

        if epoch % 1 == 0:
            print()
            trainingLog_handle.write('\n')
            torch.cuda.empty_cache()
            # tr_top1, tr_adj = test(readabilityModel, train_loader, "train")
            # torch.cuda.empty_cache()
            test_top1, test_adj = test(readabilityModel, test_loader, "test", trainingLog_handle)
            torch.cuda.empty_cache()
            print()
            trainingLog_handle.write('\n')
            if test_top1 >= best_test:
                path = './model_save/' + str(annealing_factor) + '/test_' + str(epoch) + '_t_' + str(test_top1)[
                                                                                                 0:8] + '.pth'
                torch.save(readabilityModel, path)
                best_test = test_top1

        if config.schedule:
            scheduler.step()

    trainingLog_handle.close()


def test(readabilityModel, loader, dataset_type, trainingLog_handle):
    correct_top1 = 0.0
    correct_adj = 0.0
    total = 0.0
    y_predict = []
    y_true = []

    for i, data in enumerate(loader, 1):

        input_ids, attention_mask, input_ids_sentence, attention_mask_sentence, feature, label = data
        input_ids = Variable(input_ids)
        attention_mask = Variable(attention_mask)
        input_ids_sentence = Variable(input_ids_sentence)
        attention_mask_sentence = Variable(attention_mask_sentence)
        feature = Variable(feature)
        label = Variable(torch.tensor([*label]))
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        feature = feature.cuda()
        label = label.cuda()
        input_ids_sentence = input_ids_sentence.cuda()
        attention_mask_sentence = attention_mask_sentence.cuda()

        with torch.no_grad():
            class_out, _, _, _, _, _, _, _, _, _, _, _ = readabilityModel(
                (input_ids, attention_mask, input_ids_sentence, attention_mask_sentence, feature))

        _, predicted = torch.sort(class_out.data, dim=1, descending=True)
        y_array = label.cpu().numpy().tolist()
        y_pred_array = predicted.cpu().numpy().tolist()
        y_pred = []
        for i, p in enumerate(y_pred_array):
            y_pred.append(p[0])

        y_predict.extend(y_pred)
        y_true.extend(y_array)
        total += label.size(0)
        for i, p in enumerate(y_pred_array):
            if label[i] == p[0]:
                correct_top1 = correct_top1 + 1
            if label[i] == p[0] or label[i] - p[0] == 1 or label[i] - p[0] == -1:
                correct_adj = correct_adj + 1
    precision_scores = precision_score(y_true, y_predict, average='weighted')
    recall_scores = recall_score(y_true, y_predict, average='weighted')
    f1_scores = f1_score(y_true, y_predict, average='weighted')
    cohen_kappa_scores = cohen_kappa_score(y_true, y_predict, weights='quadratic')
    print(dataset_type + '       C_Acc: %.3f%%' % (100 * correct_top1 / total) + '       Acc_adj: %.3f%%' % (
            100 * correct_adj / total) + '      Precision: %.3f%%' % (
                  precision_scores * 100) + '    Recall: %.3f%%' % (
                  recall_scores * 100) + '     F1: %.3f%%' % (
                  f1_scores * 100) + '     QWK: %.3f%%' % (
                  cohen_kappa_scores * 100))
    print('\n')
    trainingLog_handle.write(
        dataset_type + '       C_Acc: %.3f%%' % (100 * correct_top1 / total) + '       Acc_adj: %.3f%%' % (
                100 * correct_adj / total) + '      Precision: %.3f%%' % (
                precision_scores * 100) + '    Recall: %.3f%%' % (
                recall_scores * 100) + '     F1: %.3f%%' % (
                f1_scores * 100) + '     QWK: %.3f%%' % (
                cohen_kappa_scores * 100))
    trainingLog_handle.write('\n')
    return correct_top1 / total, correct_adj / total


if __name__ == '__main__':
    fire.Fire()
