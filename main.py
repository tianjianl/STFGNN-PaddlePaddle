import time
import json
import argparse

import numpy as np
import paddle
import pgl

from pgl.utils.logger import log 
from utils import (construct_model, generate_data, masked_mae_np, masked_mape_np, masked_mse_np)
from data_loader.data_utils import data_gen_mydata, gen_batch

def main(args):
    
    with open(args.config, 'r') as f:
        config = json.loads(f.read())
    
    print(json.dumps(config, sort_keys=True, indent=4))
    
    model = construct_model(config)

    batch_size = config['batch_size']
    num_of_vertices = config['num_of_vertices']
    graph_signal_matrix_filename = config['graph_signal_matrix_filename']
    n_his = config['n_his']
    n_pred = config['n_pred']
    
    if config['use_gpu'] == True:
        paddle.set_device("gpu")    

    loaders = []
    true_values = []
    

    """
    for idx, (x, y) in enumerate(generate_data(graph_signal_matrix_filename)):
        if args.test:
            x = x[:100]
            y = y[:100]
        y = y.squeeze(axis=-1)
        print(x.shape, y.shape)
        loaders.append(
            paddle.io.BatchSampler(
                x, y if idx == 0 else None,
                batch_size=batch_size,
                shuffle=(idx == 0),
                label_name='label'
            )
        )
        if idx == 0:
            training_samples = x.shape[0]
        else:
            true_values.append(y)

    train_loader, val_loader, test_loader = loaders
    val_y, test_y = true_values

    global_epoch = 1
    global_train_steps = training_samples // batch_size + 1
    all_info = []
    epochs = config['epochs']

    mod = mx.mod.Module(
        net,
        data_names=['data'],
        label_names=['label'],
        context=ctx
    )

    mod.bind(
        data_shapes=[(
            'data',
            (batch_size, config['points_per_hour'], num_of_vertices, 1)
        ), ],
        label_shapes=[(
            'label',
            (batch_size, config['points_per_hour'], num_of_vertices)
        )]
    )
    """
    
    #lr_sch = mx.lr_scheduler.PolyScheduler(
    #    max_update=global_train_steps * epochs * config['max_update_factor'],
    #    base_lr=config['learning_rate'],
    #    pwr=2,
    #    warmup_steps=global_train_steps
    #)
    #setting lr and optimizers 
    opt = config['optimizer']
    lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=config['learning_rate'], decay_steps=20, verbose=True)
    if opt == 'RMSProp':
        optim = paddle.optimizer.RMSProp(learning_rate=lr, parameters=model.parameters())
    elif opt == 'Adam':
        optim = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())
    


    num_of_parameters = 0
    trainable = 0
    nontrainable = 0

    for p in model.parameters():
        mulValue = np.prod(p.shape)  
        num_of_parameters += mulValue 
        if p.stop_gradient:
            nontrainable += mulValue  
        else:
            trainable += mulValue  
    
    print('total parameters: %s, trainable parameters: %s, nontrainbale parameters: %s' %(total, trainable, nontrainable))
    lowest_val_loss = 1e6
    for epoch in range(epochs):
        t = time.time()
        acc_list = []
        for idx, x_batch in enumerate(train_loader):
            #shape of x_batch (B, n_his+n_pred, N, 1)
            x = np.array(x_batch[:, :n_his, :, :], dtype=np.int32)
            y = np.array(x_batch[:, n_his:, :, :], dtype=np.int32)
            graph = gf.build_graph(x)
            graph.tensor()
            #model takes it input in the form of (B, n_his+n_pred, N, 1) to generate an pred array of (n_his, num_class) and the loss
            #the training accuracy of one epoch is by calculating the mean of the prediction of every batch 
            
            yhat, loss = model(graph, x)
            #shape of y is (B, n_pred, N, 1)
            #shape of yhat is (B, n_pred, N, 1)
            acc = calc_acc(y, yhat)
            acc_list.append(acc)
            loss.backward()
            
            optim.minimize(loss)
            optim.clear_grad()
            if idx % 5 == 0:
                print(epoch, idx, loss)
        
        
        print('training: Epoch: %s, ACC: %.4f, time: %.2f' % (epoch, np.mean(acc_list),time.time() - t))
        #The entrire val set is view as a single batch, batch_size is fixed to be equal to length of val set and test set
        x_val = 1 
        y_val = 1
        graph = gf.build_graph(x_val)
        graph.tensor()
        y_hat_val, loss_val = model(graph, x_val)
        acc_val = calc_acc(y_val, y_hat_val) 
        print('validation: Epoch: %s, ACC: %.4f, loss: %.4f, time: %.2f' % (epoch, acc_val, loss_val, time.time() - t))

        if loss_val < lowest_val_loss:
            x_test = 1 
            y_test = 1
            graph = gf.build_graph(x_test)
            graph.tensor()
            y_hat_test, loss_test = model(graph, x_test)
            acc_test = calc_acc(y_test, y_hat_test)

            print('test: Epoch: %s, ACC: %.4f, loss: %.4f, time: %.2f' % (epoch, acc_test, loss_test, time.time() - t))
            lowest_val_loss = loss_val

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='configuration file')
    parser.add_argument("--test", action="store_true", help="test program")
    parser.add_argument("--plot", help="plot network graph", action="store_true")
    parser.add_argument("--save", action="store_true", help="save model")
    args = parser.parse_args()
    log.info(args)
    config_filename = args.config

    main(args)

