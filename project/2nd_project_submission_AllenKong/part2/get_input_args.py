import argparse

def get_input_args_train():
    '''
    --save_dir save_directory
    --arch "vgg16"
    --learning_rate 0.01 --hidden_units 512 --epochs 20
    --gpu'''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', type = str, help = 'path to flower')
    parser.add_argument('--save_dir', type = str, help = 'path to save the checkpoint')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'CNN')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'default 0.001 LR')
    parser.add_argument('--hidden_units', type = str, default = '256,128', help = 'provide a list of 2 hidden_units')
    parser.add_argument('--epochs', type = int, default = 1, help = 'epochs')
    parser.add_argument('--gpu', action='store_true', help = 'gpu activate')
    
    in_args = parser.parse_args()
    return in_args

def get_input_args_predict():
    '''
    --dir path to the folder of flower images, 
    --input model pth,
    --topk topk,
    --category_names category_names path,
    --gpu 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='', help='path to the data directory')
    parser.add_argument('input', type = str, help = 'path to the folder of flower images')
    parser.add_argument('checkpoint', type = str, help = 'model pth')
    parser.add_argument('--topk', type = int, default = 5, help = 'topk')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'category_names')
    parser.add_argument('--gpu', action='store_true', help = 'gpu activate')
    
    in_args = parser.parse_args()
    return in_args