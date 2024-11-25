from configs import get_config
from solver import Solver
from data_loader import get_loader


if __name__ == '__main__':

    train_config = get_config(mode='train')
    test_config = get_config(mode='test')

    print(train_config)
    print(test_config)
    print('split_index:', train_config.split_index)

    train_loader = get_loader(train_config.mode, train_config.video_type ,train_config.split_index)
    test_loader = get_loader(test_config.mode,test_config.video_type, test_config.split_index)
    solver = Solver(train_config, train_loader, test_loader) # <--------

    # solver.build()        # Init'in içine dahil edildi
    #solver.evaluate(-1)	# evaluates the summaries generated using the initial random weights of the network 
    solver.train()