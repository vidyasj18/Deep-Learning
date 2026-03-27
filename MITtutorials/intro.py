## Dense layer from scratch

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self,input_dim,output_dim):
        super(MyDenseLayer,self).__init__()

        ## initialise weights and bias
        self.w = self.add_weight([input_dim,output_dim])
        self.b = self.add_weight([1,output_dim])

    
    def call(self, inputs):
        # forward propagate the inputs
        z = tf.matmul(inputs,self.w) + self.b

        # feed through non linear activation
        output = tf.math.sigmoid(z)
        return output
    


# Multi output perception
import TensorFlow as tf

layer = tf.keras.layers.Dense(units=2)


# 