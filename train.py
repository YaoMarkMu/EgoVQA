import os
import argparse
import json
import numpy as np
import random

import dataset as dt
from embed_loss import MultipleChoiceLoss
from attention_module import *
from util import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def getInput(vgg_feats, c3d_feats, questions, answers, question_lengths, max_sequence_length=30, encode_answer=True):


    if encode_answer:
        answers = np.array(answers).astype(np.int64)

    video_lengths = []
    vgg, c3d = [], []

    for i in range(len(vgg_feats)):
        vgg_feat, c3d_feat = vgg_feats[i], c3d_feats[i]
        vid_len = vgg_feat.shape[0]
        #print vgg_feat.shape, c3d_feat.shape, '===',
        if vid_len>=max_sequence_length*2:
            ss = vid_len // max_sequence_length
            vgg_feat = vgg_feat[::ss,:]
            c3d_feat = c3d_feat[::ss, :]
            #print vgg_feat.shape, c3d_feat.shape,'--111'
        elif vid_len>max_sequence_length:
            ss = random.randint(0,vid_len-max_sequence_length-1)
            vgg_feat = vgg_feat[ss:ss+max_sequence_length, :]
            c3d_feat = c3d_feat[ss:ss+max_sequence_length, :]
            #print vgg_feat.shape, c3d_feat.shape, '--222'
        else:
            #print vgg_feat.shape, c3d_feat.shape, '--333'
            pass

        video_lengths.append(vgg_feat.shape[0])
        vgg.append(torch.from_numpy(vgg_feat).to(device))
        c3d.append(torch.from_numpy(c3d_feat).to(device))

    #video_lengths = [ vgg_feat.shape[0] for vgg_feat in vgg] #[vgg.shape[1]]*bsize
    #vgg = [torch.from_numpy(vgg_feat).to(device) for vgg_feat in vgg]
    #c3d = [torch.from_numpy(c3d_feat).to(device) for c3d_feat in c3d]

    question_words = []
    for question_as in questions:
        #print question_as
        question_as = np.array(question_as).astype(np.int64)
        #print question_as.shape
        question_as = torch.from_numpy(question_as).to(device)
        question_words.append(question_as)

    if encode_answer:
        answers = torch.from_numpy(answers).to(device,non_blocking=True)
        #answers = answers.view(answers.size(0),1)
        
    #video_features = torch.cat([c3d,vgg],dim=2)
    #video_features = video_features.view(video_features.size(0),video_features.size(1),1,1,video_features.size(2))
    video_features = [vgg, c3d]

    data_dict = {}
    data_dict['video_features'] = video_features
    data_dict['video_lengths'] = video_lengths
    data_dict['question_words'] = question_words
    data_dict['answers'] = answers
    data_dict['question_lengths'] = question_lengths
    data_dict['num_mult_choices'] = 5

    return data_dict
            

def main():
    """Main script."""
    torch.manual_seed(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='train/test')
    parser.add_argument('--save_path', type=str, default='./saved_models/' ,
                        help='path for saving trained models')     
    parser.add_argument('--split', type=int, help='which of the three splits to train/val/test, option: 0 | 1 | 2')
    parser.add_argument('--test', type=int, default=0, help='0 | 1')
    parser.add_argument('--memory_type', type=str, help='0 | 1')

    args = parser.parse_args()
    
    
    args.dataset = 'ego_vqa'
    args.word_dim = 300
    args.vocab_num = 4000
    args.pretrained_embedding = 'data/word_embedding.npy'
    args.video_feature_dim = 4096
    args.video_feature_num = 20
    args.memory_dim = 256
    args.batch_size = 8
    args.reg_coeff = 1e-5
    args.learning_rate = 0.001
    args.preprocess_dir = 'data'
    args.log = './logs'
    args.hidden_size = 256

    video_cam = ['1_D', '1_M', '2_F', '2_M', '3_F', '3_M', '4_F', '4_M',
                 '5_D', '5_M', '6_D', '6_M', '7_D', '7_M', '8_M', '8_X']


    with open('./data/data_split.json', 'r') as f:
        splits = json.load(f)
    assert args.split < len(splits)
    video_cam_split = splits[args.split]

    dataset = dt.EgoVQA(args.batch_size, './data', './data/feats', video_cam_split)
    

    #args.memory_type = '_mrm2s'   # HME-QA: see HME-QA paper
    #args.memory_type = '_stvqa'  # st-vqa: see TGIF-QA paper
    #args.memory_type = '_enc_dec' # plain LSTM
    #args.memory_type = '_co_mem' # Co-Mem
    # TODO: if possible, add new algorithm
    # TODO: analyze different types of questions

    args.image_feature_net = 'concat'
    args.layer = 'fc'

    args.save_model_path = args.save_path + 'model_%s_%s%s' % (args.image_feature_net,args.layer,args.memory_type)
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    if not os.path.exists(os.path.join(args.save_model_path,'s%d'%(args.split))):
        os.makedirs(os.path.join(args.save_model_path,'s%d'%(args.split)))

    args.pretrain_model_path = './pretrain_models/' + 'model_%s_%s%s' % (args.image_feature_net,args.layer,args.memory_type)

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


    if args.memory_type=='_mrm2s':
        rnn = AttentionTwoStream(feat_channel, feat_dim, text_embed_size, args.hidden_size,
                             voc_len, num_layers, word_matrix, max_len=max_sequence_length)
        rnn = rnn.cuda()

        best_pretrain_state = 'rnn-5100-vl_80.451-t_81.002.pkl'

    elif args.memory_type=='_stvqa':
        feat_channel *= 2
        rnn = TGIFBenchmark(feat_channel, feat_dim, text_embed_size, args.hidden_size,
                                 voc_len, num_layers, word_matrix, max_len=max_sequence_length)
        rnn = rnn.cuda()

        best_pretrain_state = 'rnn-0400-vl_69.603-t_68.562.pkl'

    elif args.memory_type=='_enc_dec':
        feat_channel *= 2
        rnn = LSTMEncDec(feat_channel, feat_dim, text_embed_size, args.hidden_size,
                                 voc_len, num_layers, word_matrix, max_len=max_sequence_length)
        rnn = rnn.cuda()

        best_pretrain_state = 'rnn-1200-vl_66.518-t_65.817.pkl'

    elif args.memory_type=='_co_mem':
        rnn = CoMemory(feat_channel, feat_dim, text_embed_size, args.hidden_size,
                                 voc_len, num_layers, word_matrix, max_len=max_sequence_length)
        rnn = rnn.cuda()

        best_pretrain_state = 'rnn-1200-vl_76.516-t_75.708.pkl'

    else:
        assert 1==2

    #################################
    # load pretrain model to finetune
    #################################

    pretrain_path = os.path.join('./pretrain_models',
                                 'model_%s_%s%s' % (args.image_feature_net,args.layer,args.memory_type),
                                 best_pretrain_state)

    if os.path.isfile(pretrain_path):
        #rnn.load_state_dict(torch.load(pretrain_path))
        print 'Load from ', pretrain_path
    else:
        print 'Cannot load ', pretrain_path


    # loss function
    criterion = MultipleChoiceLoss(margin=1, size_average=True).cuda()

    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.learning_rate, weight_decay=0.0005)

    best_test_acc = 0.0
    best_test_iter = 0
    
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


        if epoch>=0:
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

                    vgg, c3d, questions, answers, question_lengths, _,_ = dataset.get_test_example()
                    data_dict = getInput(vgg, c3d, questions, answers, question_lengths)
                    outputs, predictions = rnn(data_dict)
                    targets = data_dict['answers']

                    acc = rnn.accuracy(predictions, targets)
                    accuracy.update(acc.item(), len(vgg))


                test_acc = accuracy.avg
                print('Test iter %d, acc %.3f' % (iter, accuracy.avg))
                dataset.reset_test()


                if best_test_acc < accuracy.avg:
                    best_test_acc = accuracy.avg
                    best_test_iter = iter

                print('[Test] iter %d, acc %.3f, best acc %.3f at iter %d' % (iter,test_acc,best_test_acc,best_test_iter))

                torch.save(rnn.state_dict(), os.path.join(args.save_model_path, 's%d' % (args.split,), 'rnn-%04d-vl_%.3f-t_%.3f.pkl' %(iter,val_acc,test_acc)))
                rnn.train()

            
            


if __name__ == '__main__':
    main()
