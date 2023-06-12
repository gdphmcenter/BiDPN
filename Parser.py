import argparse

def Parser():
    parser = argparse.ArgumentParser(description='CWRU_fault Predition parameter')
    parser.add_argument('-bap', '--base_path', type=str, help='Base folder in Linux', default='/home/amax/ybin/CWRU_ Algorithm1218/Eucli_Three_Model_1218')
    parser.add_argument('-dap', '--data_path', type=str, help='Dataset folder in Linux', default='/home/amax/ybin/CWRU_Fewshot_Data')
    parser.add_argument('-rp', '--raw_path', type=str, help='Raw CWRU dataset folder in Linux', default='/home/amax/ybin/ybin_data/CWRU_CWT')
    parser.add_argument('-dana', '--dataset_name', type=str, help='Dataset classification in train', default='12+48DE+Normal')
    parser.add_argument('-TFw', '--TiFre_graph_weight', type=float, help='the contribution of time_frequency graph for predition', default=0.3)
    parser.add_argument('-Tw', '--Time_weight', type=float, help='the contribution of time field for predition', default=0.5)
    parser.add_argument('-Fw', '--Frequency_weight', type=float, help='the contribution of frequency for predition', default=0.2)
    parser.add_argument('-kt', '--k_train', type=int, help='number of classes in train', default=3)
    parser.add_argument('-nt', '--n_train', type=int, help='number of support sample in each class in train', default=1)
    parser.add_argument('-qt', '--q_train', type=int, help='number of query sample in each class in train', default=1)
    parser.add_argument('-kv', '--k_val', type=int, help='number of classes in val or test', default=3)
    parser.add_argument('-nv', '--n_val', type=int, help='number of support sample in each class in val or test', default=1)
    parser.add_argument('-qv', '--q_val', type=int, help='number of query sample in each class in val or test', default=1)
    parser.add_argument('-epo', '--epochs', type=int, help='number of epoch in model train', default= 500)
    parser.add_argument('-epi', '--episodes_per_epoch', type=int, help='number of episode in an epoch', default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate for the model optimize', default=1e-3)#1e-4
    parser.add_argument('-lrs', '--lr_scheduler_step', type=int, help='StepLR learning rate schedule step', default=20)
    parser.add_argument('-lrg', '--lr_scheduler_gamma', type=float, help='StepLR learning rate schedule gamma', default=0.5)
    parser.add_argument('-ms', '--manual_seed', type=int, help='Initialization of manual seed', default=10)

    return parser

