import os
import argparse
import json
import numpy as np
import dataset as dt
from attention_module import *
from util import AverageMeter
from train import getInput
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
 


def main():
    """Main script."""
    torch.manual_seed(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./saved_models/' ,
                        help='path for saving trained models')     
    #parser.add_argument('--split', type=int, help='which of the three splits to train/val/test, option: 0 | 1 | 2')
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

    with open('./data/data_split.json', 'r') as f:
        splits = json.load(f)

    #args.memory_type = '_mrm2s'
    args.image_feature_net = 'concat'
    args.layer = 'fc'


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

    elif args.memory_type=='_stvqa':
        feat_channel *= 2
        rnn = TGIFBenchmark(feat_channel, feat_dim, text_embed_size, args.hidden_size,
                                 voc_len, num_layers, word_matrix, max_len=max_sequence_length)
        rnn = rnn.cuda()

    elif args.memory_type=='_enc_dec':
        feat_channel *= 2
        rnn = LSTMEncDec(feat_channel, feat_dim, text_embed_size, args.hidden_size,
                                 voc_len, num_layers, word_matrix, max_len=max_sequence_length)
        rnn = rnn.cuda()

    elif args.memory_type=='_co_mem':
        rnn = CoMemory(feat_channel, feat_dim, text_embed_size, args.hidden_size,
                                 voc_len, num_layers, word_matrix, max_len=max_sequence_length)
        rnn = rnn.cuda()

    else:
        assert 1==2
    



    count_type = defaultdict(int)
    count_correct = defaultdict(int)
    accuracy = AverageMeter()
    correct = 0
    split_acc = []
    split_acc2 = []

    use_pretrain = False

    for sp in [0, 1, 2]:
        video_cam_split = splits[sp]
        dataset = dt.EgoVQA(args.batch_size, './data', './data/feats', video_cam_split)

        if use_pretrain:
            rnn.load_state_dict(torch.load('pretrain_models/model_concat_fc_stvqa/rnn-0400-vl_69.603-t_68.562.pkl'))
        else:
            args.save_model_path = os.path.join(args.save_path, 'model_%s_%s%s' %
                                                (args.image_feature_net, args.layer,
                                                 args.memory_type),
                                                's%d' % (sp))

            files = sorted(os.listdir(args.save_model_path))
            prefix = 'rnn-'
            max_val = 0
            max_test = 0
            max_iter = -1
            best_state = None
            for f in files:
                if f.startswith(prefix):
                    segs = f.split('-')
                    it0 = int(segs[1])
                    val_acc = float(segs[2][3:])
                    test_acc = float(segs[3][2:-4])
                    if val_acc > max_val:
                        max_iter = it0
                        max_val = val_acc
                        max_test = test_acc

                        best_state = f
            print 'load checkpoint', best_state, max_val, max_test
            assert best_state is not None
            split_acc.append(max_test)
            rnn.load_state_dict(torch.load(os.path.join(args.save_model_path,best_state)))


        with torch.no_grad():
            rnn.eval()
            correct_sp = 0
            idx = 0
            while dataset.has_test_example:
                if idx%100==0:
                    print 'Test iter %d/%d' % (idx,dataset.test_example_total)
                idx += 1

                vgg, c3d, questions, answers, question_lengths, qtext, atext = dataset.get_test_example()

                data_dict = getInput(vgg, c3d, questions, answers, question_lengths)
                outputs, predictions = rnn(data_dict)
                targets = data_dict['answers']

                acc = rnn.accuracy(predictions, targets)
                accuracy.update(acc.item(), len(vgg))


                prediction = predictions.item()
                target = targets.item()

                tp = None
                qtext = qtext.lower()
                if 'am i' in qtext and 'doing' in qtext:
                    # first-person action
                    tp = '1_action_1st'
                elif 'doing' in qtext:
                    # third-person action
                    tp='2_action_3rd'
                elif 'how many' in qtext:
                    tp = '5_count'
                elif 'color' in qtext and 'what' in qtext:
                    tp = '6_what_color'
                elif 'what' in qtext:
                    if 'am i' in qtext or 'my' in qtext:
                        tp = '3_what_obj_1st'
                    else:
                        tp = '3_what_obj_3rd'
                elif qtext.startswith('who'):
                    if 'me' in qtext or 'am i' in qtext:
                        tp = '4_who_1st'
                    else:
                        tp = '4_who_3rd'
                else:
                    tp = '7_other'

                #print tp, ' ', qtext, ' ', atext, prediction

                count_type[tp] += 1
                if prediction == target:
                    correct += 1
                    correct_sp += 1
                    count_correct[tp] += 1


            test_acc = 1.0*correct_sp / dataset.test_example_total
            print correct, dataset.test_example_total
            print('Test acc %.3f' % (test_acc,))
            split_acc2.append(test_acc)

    class_acc = 0.0
    print 'Question types accuracy: ',
    for tp in sorted(count_type.keys()):
        a = count_correct[tp]
        b = count_type[tp]
        if b==0:
            b=1
        print '(',tp,a,b,1.0*a/b,') '
        class_acc += 1.0*a/b
    print
    print 'Question types accuracy: ',
    for tp in sorted(count_type.keys()):
        a = count_correct[tp]
        b = count_type[tp]
        if b == 0:
            b = 1
        print round(100.0*a/b,2), '& ',

    print
    print count_type
    print 'Question types per class accuracy: ', class_acc / len(count_type.keys())

    print 'split acc', split_acc, split_acc2
            
            


if __name__ == '__main__':
    main()
