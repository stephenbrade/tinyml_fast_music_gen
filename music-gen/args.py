import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', action = 'store', type = str, required = True, dest = 'exp_name')
    parser.add_argument('--seed', action = 'store', type = int, dest = 'seed', default = 0)
    
    parser.add_argument('--eval', action = 'store_true', dest = 'eval')
    parser.add_argument('--target_model', action = 'store', type = str, dest = 'target_model', default = 'lstm', choices = ['music-trans'])
    parser.add_argument('--student_model', action = 'store', type = str, dest = 'student_model', default = 'LSTM', choices = ['Music-LSTM', 'Music-Trans'])

    parser.add_argument('--num_workers', action = 'store', type = int, dest = 'num_workers', default = 4)
    parser.add_argument('--batch_size', action = 'store', type = int, dest = 'batch_size', default = 64)

    parser.add_argument('--num_epochs', action = 'store', type = int, dest = 'num_epochs', default = 10)
    parser.add_argument('--lr', action = 'store', type = float, dest = 'lr', default = 1e-3)
    parser.add_argument('--grad_acc', action = 'store', type = int, dest = 'accumulation', default = 1)

    parser.add_argument('--embedding_dim', action = 'store', type = int, dest = 'embedding_dim', default = 768)
    parser.add_argument('--hidden_dim', action = 'store', type = int, dest = 'hidden_dim', default = 768)
    parser.add_argument('--num_layers', action = 'store', type = int, dest = 'num_layers', default = 3)
    parser.add_argument('--fc_dim', action = 'store', type = int, dest = 'fc_dim', default = 1024)

    parser.add_argument('--rep_sim', action = 'store_true', dest = 'rep_sim')
    parser.add_argument('--repdist', action = 'store', type = str, dest = 'rep_dist', default = 'CKA')
    parser.add_argument('--alpha', action = 'store', type = float, dest = 'rep_sim_alpha', default = 1.0)
    parser.add_argument('--untrained', action = 'store_false', dest = 'pretrained')
    parser.add_argument('--noise', action = 'store_true', dest = 'use_noise')
    
    parser.add_argument('--logging', action = 'store', type = str, dest = 'logging', default = 'logs')

    parser.add_argument('--multi_gpu', action = 'store_true', dest = 'multi_gpu')
    parser.add_argument('--load_ckpt', action = 'store_true', dest = 'load_ckpt')
    
    parser.set_defaults(rep_sim = False, pretrained = True, eval = False, load_ckpt = False)
    args = parser.parse_args()

    return args