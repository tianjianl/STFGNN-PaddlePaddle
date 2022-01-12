import paddle 
import paddle.nn as nn 
import paddle.nn.functional as func
import numpy as np
class STFGNNModel(nn.Layer):
    """implementation of STFGNN in https://arxiv.org/abs/2012.09641"""
    def __init__(self, config, mat):
        super(STFGNNModel, self).__init__()
        self.config = config
        self.n_pred = config['num_for_predict']
        self.layers = len(config['filters'])
        self.graph_linear = nn.Linear(32, 64)
        self.conv2ds_left = nn.LayerList([nn.Conv2D(32, 32, [4,1], data_format='NHWC') for _ in range(self.layers)])
        self.conv2ds_right = nn.LayerList([nn.Conv2D(32, 32, [4,1], data_format='NHWC') for _ in range(self.layers)])
        self.output_linears_1 = nn.LayerList([nn.Linear(32*3, 32) for _ in range(self.n_pred)])
        self.output_linears_2 = nn.LayerList([nn.Linear(32, 5) for _ in range(self.n_pred)])
        self.fusion = paddle.to_tensor(mat, dtype=paddle.float32)
        self.emb = nn.Embedding(7, 32)
    
    
    
    def graph_mul_block(self, data, fusion, activation):
        
        assert activation in {'glu', 'relu'}
        
        #shape of fusion is (4N, 4N)
        #shape of data is (4N, B, C)
        _, B, C = data.shape
        data = paddle.flatten(data, start_axis=1, stop_axis=2)
        data = paddle.matmul(fusion, data)
        data = paddle.reshape(data, shape=[-1, B, C])
        #(4N, B, C)

        data = self.graph_linear(data)
    
        if activation == 'glu':
            left, right = paddle.split(data, num_or_sections=2, axis=2)
            return left + func.sigmoid(right)
        elif activation == 'relu':
            return func.relu(data)



    def STFGN_module(self, data, fusion, activation):
        """
        multiple GCN layers with cropping and max pooling
        ________________________________________________

        data: paddle tensor of shape (4N, B, C)

        fusion: STF matrix, paddle tensor of shape (4N, 4N)

        activation: string of relu/glu

        returns a tensor of (N, B, C')

        """
        
        need_concat = []
        for i in range(len(self.config['filters'])):
            data = self.graph_mul_block(data, fusion, activation)
            need_concat.append(data)
	

        # shape of each element is (1, N, B, C'), concat into (L, N, B, C') then selected max to get (N, B, C')

        N, B, C_prime = int(data.shape[0]/4), data.shape[1], 32

        need_concat = [paddle.reshape(i[N:2*N,:,:], [1, N, B, C_prime]) for i in need_concat] 
        need_concat = paddle.concat(need_concat) #(L, N, B, C')
        return paddle.max(need_concat, axis=0)  #(N, B, C')

    
    def STFGN_layer(self, data, fusion, activation, num):

        '''
        mutiple STFGN_modules + Gated CNN = STFGN Layer    
        Parameters
        ----------
        data: tensor, shape is (B, T, N, C)

        fusion: tensor, shape is (4N, 4N)

        activation: str, choice of 'glu' or 'relu'

        temporal_emb, spatial_emb: bool

        Returns
        ----------
        output shape is (B, T-3, N, C')
        '''
        
        conv2d_left = self.conv2ds_left[num]
        conv2d_right = self.conv2ds_right[num]
        # shape is (B, T, N, C)
        #data = position_embedding(data, T, num_of_vertices, num_of_features,
        #                          temporal_emb, spatial_emb,
        #                          prefix="{}_emb".format(prefix))
   
        # Gated CNN 
        c_prime = len(self.config['filters'])
        if len(data.shape) == 5:
            data = paddle.flatten(data, start_axis=3, stop_axis=4)
        data_left = func.sigmoid(conv2d_left(data))
        data_right = paddle.tanh(conv2d_right(data))
        data_time_axis = paddle.multiply(data_left, data_right)  

        # shape is (B, T-3, N, C)
        data_res = data_time_axis
        _, T, _, _ = data.shape
        need_concat = []
        for i in range(T - 3):
            # shape is (B, 4, N, C)
            t = data[:, i:i+4, :, :]

            # shape is (B, 4N, C)
            t = paddle.flatten(t, start_axis=1, stop_axis=2)

            # shape is (4N, B, C)
            t = paddle.transpose(t, (1, 0, 2))

            # shape is (N, B, C')
            t = self.STFGN_module(t, fusion, activation=activation)

            # shape is (B, N, C')
            t = paddle.transpose(t, (1, 0, 2))

            # shape is (B, 1, N, C')
            t = paddle.reshape(t, [t.shape[0], 1, t.shape[1], t.shape[2]])

            need_concat.append(t)

            # shape is (B, T-3, N, C')
        after_concat = paddle.concat(need_concat, axis=1)

        if self.config['gated_cnn'] == 'True':
            return after_concat + data_res
        else:
            return after_concat
   

    def output_layer(self, data, num):
        '''
        Parameters
        ----------
        data: paddle tensor, shape is (B, T, N, C)

        Returns
        ----------
        padddle tensor of shape (B, N, 1)

        The output_layer outputs one prediction 
        '''

        linear1 = self.output_linears_1[num]
        linear2 = self.output_linears_2[num]
        # data shape is (B, N, T', C)
        data = paddle.transpose(data, (0, 2, 1, 3))
        # (B, N, T' * C)
        input_length = self.config['num_of_history']
        data = paddle.flatten(data, start_axis=2, stop_axis=3)
        # (B, N, C')
        data = func.relu(linear1(data))

        # (B, N, num_class)
        data = linear2(data)
        return data

    def forward(self, data, label):
        predlen = self.config['num_for_predict']
        fusion = self.fusion
        activation = self.config['act_type']
        data = paddle.to_tensor(data)
        label = np.expand_dims(label, axis=-1)
        label = paddle.to_tensor(label)
        #label shape B 12 N 1
        label = label[:, :predlen, :, :]
        label = paddle.transpose(label, (0, 2, 1, 3))

        data = self.emb(data)
        for i in range(len(self.config['filters'])):
            data = self.STFGN_layer(data, fusion, activation, i)
        
        need_concat = []
        for i in range(self.config['num_for_predict']):
            need_concat.append(self.output_layer(data, i))
        
        #(B , N, predlen * numclass)
        y_hat = paddle.concat(need_concat, axis=1)
        #(B, N, predlen, numclass)
        #label shape (B, N, predlen, 1)

        y_hat = paddle.reshape(y_hat, (y_hat.shape[0], y_hat.shape[1], self.config['num_for_predict'], 5))
        label = paddle.cast(label, 'int64')
        loss, ysoft = func.softmax_with_cross_entropy(logits=y_hat, label=label, return_softmax=True)
        loss = paddle.mean(loss)
        #ysoft shape is (B, N, predlen, 5)
        return loss, [paddle.argmax(ysoft[:, :, i, :], axis=-1) for i in range(predlen)]
