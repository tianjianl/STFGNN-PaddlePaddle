import paddle 
import paddle.nn as nn 
import paddle.nn.functional as func

class STFGNNModel(nn.Layer):
    """implementation of STFGNN in https://arxiv.org/abs/2012.09641"""
    def __init__(self, args):
        super(STFGNNModel, self).__init__()
        self.args = args
    
    
    def graph_mul_block(self, data, fusion, activation):
    """
    graph multiplication block
    _________________________
    
    data: paddle tensor, input data shape of (4N, B, C)

    fusion: paddle tensor, STF matrix shape of (4N, 4N)

    returns: paddle tensor, shape of (4N, B, C')
   
    """
        assert activation in {'glu', 'relu'}
        data = paddle.matmul(fusion, data)
        #(4N, B, C)

        data = self.graph_linear(data)
    
        if activation == 'glu':
            left, right = paddle.split(data, num_or_sections=2, axis=2)
            return left + func.sigmoid(right)
        elif activation == 'relu'
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
        for i in range(len(self.args.blocks)):
            data = graph_mul_block(data, fusion, activation)
            need_concat.append(data)
	

        # shape of each element is (1, N, B, C'), concat into (L, N, B, C') then selected max to get (N, B, C')

    	N, B, C_prime = int(data.shape[0]/4), data.shape[1], len(filers)
        need_concat = [paddle.expand(i[N:2*N,:,:], [1, N, B, C_prime] for i in need_concat] 
        need_concat = paddle.concat(need_concat) #(L, N, B, C')
        return paddle.max(need_concat, axis=0)  #(N, B, C')

    
    def STFGN_layer(self, data, fusion, activation)

        """
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
        """

        # shape is (B, T, N, C)
        data = position_embedding(data, T, num_of_vertices, num_of_features,
                                  temporal_emb, spatial_emb,
                                  prefix="{}_emb".format(prefix))
   
        # Gated CNN 
        # shape of data temp is (B, C, N, T)
        c_prime = len(filters)
        data_temp = paddle.transpose(data, (0, 3, 2, 1))
        data_left = func.sigmoid(self.conv2d_left(data_temp))
        data_right = paddle.tanh(self.conv2d_right(data_temp))
        data_time_axis = data_left * data_right
    

        # shape is (B, T-3, N, C)
        data_res = paddle.transpose(data_time_axis, (0, 3, 2, 1))

        need_concat = []
        for i in range(T - 3):
            # shape is (B, 4, N, C)
            t = data[:, i:i+4, :, :]
        
            # shape is (B, 4N, C)
            t = paddle.flatten(t, start_axis=1, stop_axis=2)
        
            # shape is (4N, B, C)
            t = paddle.transpose(t, (1, 0, 2))

            # shape is (N, B, C')
            t = STFGN_module(t, fusion, activation=activation)

            # shape is (B, N, C')
            t = paddle.transpose(t, (1, 0, 2))
        
            # shape is (B, 1, N, C')
            t = paddle.expand(t, [t.shape[0], 1, t.shape[1], t.shape[2]])
        
            need_concat.append(t)

            # shape is (B, T-3, N, C')
            after_concat = paddle.concat(need_concat, axis=1)
    
            if self.args.use_gated = True:
                return after_concat + data_res
            else:
                return after_concat
   

    def output_layer(self, data, num):
        """
        Parameters
        ----------
        data: paddle tensor, shape is (B, T, N, C)

        Returns
        ----------
        padddle tensor of shape (B, T', N)
        
	"""

        # data shape is (B, N, T, C)
        data = paddle.transpose(data, (0, 2, 1, 3))

        # (B, N, T * C)
        data = paddle.reshape(data, (-1, self.args.N, input_length * num_of_features))

        # (B, N, C')
        data = func.relu(self.output_linear1(data)

        # (B, N, T'=4)
        data = self.output_linear2(data)

        return data

    def forward(self, data, fusion, label):
        for i in range(self.args.lnum):
            data = STFGN_layer(data, fusion, self.args.activation, i)
        
        need_concat = []
        for i in range(self.args.predlen):
            need_concat.append(output_layer(data, i))
        
        #(B , N, predlen * numclass)
        yhat = paddle.concat(need_concat, axis=1)
        print('yhat.shape=', yhat.shape)
        #(B, N, predlen, numclass)
        #label shape (B, N, predlen, 1)
        yhat = paddle.expand(y_hat, (y_hat.shape[0], y_hat.shape[1], self.args.predlen, 5))
        loss, ysoft = func.softmax_with_cross_entropy(logits=yhat, label=label, return_softmax=True)
        loss = paddle.mean(loss)
        #ysoft shape is (B, N, predlen, 5)
        return loss, [paddle.argmax(ysoft[:, :, i, :], axis=-1) for i in range(predlen)]
        
