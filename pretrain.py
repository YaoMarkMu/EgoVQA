import os
import argparse
import json
import numpy as np
import torch
import dataset as dt
from embed_loss import MultipleChoiceLoss
from attention_module import *
from util import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def getInput(vgg_input, c3d_input, questions, answers, question_lengths, encode_answer=True):
    if encode_answer:
        answers = np.array(answers).astype(np.int64)

    video_lengths = []
    vgg = []
    c3d = []

    for vgg_feat, c3d_feat in zip(vgg_input, c3d_input):
        vgg_feat = np.array(vgg_feat).astype(np.float32)
        c3d_feat = np.array(c3d_feat).astype(np.float32)
        #print vgg_feat.shape, c3d_feat.shape
        video_lengths.append(vgg_feat.shape[0])
        vgg.append(torch.from_numpy(vgg_feat).to(device))
        c3d.append(torch.from_numpy(c3d_feat).to(device))

    question_words = []
    for question_as in questions:
        question_as = np.array(question_as).astype(np.int64)
        question_as = torch.from_numpy(question_as).to(device)
        question_words.append(question_as)

    if encode_answer:
        answers = torch.from_numpy(answers).to(device,non_blocking=True)

    video_features = [vgg, c3d]

    data_dict = {}
    data_dict['video_features'] = video_features
    data_dict['video_lengths'] = video_lengths
    data_dict['question_words'] = question_words
    data_dict['answers'] = answers
    data_dict['question_lengths'] = question_lengths
    data_dict['num_mult_choices'] = 4

    return data_dict
            

def main():
    """Main script."""
    torch.manual_seed(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='train/test')
    parser.add_argument('--save_path', type=str, default='./pretrain_models/' ,
                        help='path for saving trained models')     
    parser.add_argument('--test', type=int, default=0, help='0 | 1')
    parser.add_argument('--memory_type', type=str, help='0 | 1')

    args = parser.parse_args()
    
    
    args.dataset = 'zh_vqa'
    args.word_dim = 300
    args.vocab_num = 4000
    args.pretrained_embedding = './data_zhqa/word_embedding.npy'
    args.video_feature_dim = 4096
    args.video_feature_num = 20
    args.memory_dim = 256
    args.batch_size = 8
    args.reg_coeff = 1e-5
    args.learning_rate = 0.001
    args.preprocess_dir = 'data_zhqa'
    args.log = './logs'
    args.hidden_size = 256

    dataset = dt.ZHVQA(args.batch_size, './data_zhqa')
    args.image_feature_net = 'concat'
    args.layer = 'fc'

    args.save_model_path = args.save_path + 'model_%s_%s%s' % (args.image_feature_net,args.layer,args.memory_type)
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)



    #############################
    # get video feature dimension
    #############################
    feat_channel = args.video_feature_dim
    feat_dim = 1
    text_embed_size = args.word_dim
    voc_len = args.vocab_num
    num_layers = 2
    max_sequence_length = args.video_feature_num
    word_matrix = np.load(args.pretrained_embedding)


    if args.memory_type =='_mrm2s':
        rnn = AttentionTwoStream(feat_channel, feat_dim, text_embed_size, args.hidden_size,
                             voc_len, num_layers, word_matrix, max_len=max_sequence_length)
        rnn = rnn.to(device)

    elif args.memory_type =='_stvqa':
        feat_channel *= 2
        rnn = TGIFBenchmark(feat_channel, feat_dim, text_embed_size, args.hidden_size,
                                 voc_len, num_layers, word_matrix, max_len=max_sequence_length)
        rnn = rnn.to(device)

    elif args.memory_type =='_enc_dec':
        feat_channel *= 2
        rnn = LSTMEncDec(feat_channel, feat_dim, text_embed_size, args.hidden_size,
                                 voc_len, num_layers, word_matrix, max_len=max_sequence_length)
        rnn = rnn.to(device)

    elif args.memory_type =='_co_mem':
        rnn = CoMemory(feat_channel, feat_dim, text_embed_size, args.hidden_size,
                                 voc_len, num_layers, word_matrix, max_len=max_sequence_length)
        rnn = rnn.to(device)

    else:
        raise Exception('Please specify memory_type')

    # loss function
    criterion = MultipleChoiceLoss(margin=1, size_average=True).to(device)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.learning_rate, weight_decay=0.0005)


    iter = 0
    for epoch in range(0, 20):
        dataset.reset_train()
        
        while dataset.has_train_batch:
            iter += 1

            vgg, c3d, questions, answers, question_lengths = dataset.get_train_batch()
            data_dict = getInput(vgg, c3d, questions, answers, question_lengths)
            outputs, predictions = rnn(data_dict)
            targets = data_dict['answers']

            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        

            acc = rnn.accuracy(predictions, targets)
            print('Train iter %d, loss %.3f, acc %.2f' % (iter,loss.data,acc.item()))


            if iter % 300==0:
                rnn.eval()

                # val iterate over examples
                with torch.no_grad():

                    idx = 0
                    accuracy = AverageMeter()

                    while dataset.has_val_example:
                        if idx%10==0:
                            print 'Val iter %d/%d' % (idx,dataset.val_example_total)
                        idx += 1

                        vgg, c3d, questions, answers, question_lengths = dataset.get_val_example()
                        data_dict = getInput(vgg, c3d, questions, answers, question_lengths)
                        outputs, predictions = rnn(data_dict)
                        targets = data_dict['answers']

                        acc = rnn.accuracy(predictions, targets)
                        accuracy.update(acc.item(), len(vgg))


                    val_acc = accuracy.avg
                    print('Val iter %d, acc %.3f' % (iter, val_acc))
                    dataset.reset_val()


                    idx = 0
                    accuracy = AverageMeter()

                    while dataset.has_test_example:
                        if idx%10==0:
                            print 'Test iter %d/%d' % (idx,dataset.test_example_total)
                        idx += 1

                        vgg, c3d, questions, answers, question_lengths, _ = dataset.get_test_example()
                        data_dict = getInput(vgg, c3d, questions, answers, question_lengths)
                        outputs, predictions = rnn(data_dict)
                        targets = data_dict['answers']

                        acc = rnn.accuracy(predictions, targets)
                        accuracy.update(acc.item(), len(vgg))


                    test_acc = accuracy.avg
                    print('Test iter %d, acc %.3f' % (iter, accuracy.avg))
                    dataset.reset_test()

                    torch.save(rnn.state_dict(), os.path.join(args.save_model_path, 'rnn-%04d-vl_%.3f-t_%.3f.pkl' %(iter,val_acc,test_acc)))
                rnn.train()





if __name__ == '__main__':
    main()
