from model import *
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='baseline')
    parser.add_argument('-g', '--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    
    parser.add_argument('-s', '--save_model', dest='save_model', help='the name of saved model', 
                        default='model')
    
    parser.add_argument('-l', '--load_model', dest='load_model', help='the name of load model', 
                        default='model')                        
                                
    parser.add_argument('-b', '--batch_size', dest='batch_size', help='batch size',
                        default=16, type=int)
                        
    parser.add_argument('-e', '--epochs', dest='epochs', help='total numper of epochs',
                        default=100, type=int)
    
    parser.add_argument('-sz', '--size', dest='size', help='the size of image',
                        default=256, type=int)
    
    parser.add_argument('-q', '--quick', dest='quick', help='quick',
                        default=0, type=int)
                        
    parser.add_argument('-is', '--is_save', dest='is_save', help='save or not',
                        default=1, type=int)
#                         action='store_true')
    parser.add_argument('-df', '--df_specific', dest='df_specific', help='specific df or otherwise',
                        default=0, type=int)
    args, _ = parser.parse_known_args()
    
    return args
    
args = parse_args()

M = InceptResV2_DF_one(gpu_id=args.gpu_id, want_size=args.size, batch_size=args.batch_size, 
                        epochs=args.epochs, model_name=args.save_model, num_classes=2, 
                        print_mode='logger', quick=args.quick)

os.environ['CUDA_VISIBLE_DEVICES'] = str(M.gpu_id)
_print = M.Set_print()
_print(str(args))

sess = M.Session()
M.Build_network()
sess.run(tf.global_variables_initializer())
M.Set_save()
M.Make_dataset()
M.Make_dataset_DF_one(specific=args.df_specific)

M.Train('Accuracy')