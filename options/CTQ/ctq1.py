import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as KK
import numpy as np
num_classes=400
#import psycopg2 as p2
feature = [180, 42, 195, 6, 340, 135, 224, 254, 359, 84, 120, 109, 254, 163, 91, 78, 335, 249, 266, 57, 41, 262, 120, 369, 84, 42, 126, 214, 119, 224, 211, 200, 344, 39, 257, 28, 27, 162, 216, 147, 58, 155, 153, 86, 317, 370, 233, 303, 189, 347, 79, 390, 390, 250, 239, 243, 249, 219, 99, 127, 101, 349, 376, 91, 196, 371, 159, 116, 273, 300, 226, 122, 62, 58, 232, 288, 329, 26, 55, 201, 211, 56, 19, 273, 318, 8, 160, 100, 198, 346, 101, 8, 252, 84, 216, 210, 242, 2, 354, 214, 43, 135, 18, 190, 338, 281, 329, 190, 32, 20, 394, 267, 117, 56, 198, 161, 191, 216, 352, 58, 136, 206, 66, 119, 225, 39, 103, 163, 234, 295, 150, 25, 100, 248, 393, 276, 313, 159, 286, 185, 272, 360, 328, 373, 279, 154, 230, 309, 377, 50, 66, 52, 385, 246, 42, 77, 311, 216, 107, 359, 166, 164, 156, 331, 72, 42, 369, 378, 70, 255, 119, 131, 76, 263, 46, 355, 65, 243, 147, 231, 380, 311, 215, 201, 119, 258, 320, 183, 275, 240, 149, 309, 301, 356, 353, 199, 306, 244, 98, 192, 58, 11, 152, 360, 265, 361, 293, 170, 205, 392, 362, 30, 378, 29, 74, 35, 106, 177, 232, 221, 266, 281, 118, 209, 251, 183, 309, 223, 290, 73, 320, 89, 138, 157, 324, 284, 368, 275, 240, 203, 202, 81, 394, 166, 171, 98, 283, 138, 373, 122, 342, 299, 180, 42, 195, 6, 340, 135, 224, 254, 359, 84, 120, 109, 254, 163, 91, 78, 335, 249, 266, 57, 41, 262, 120, 369, 84, 42, 126, 214, 119, 224, 211, 200, 344, 39, 257, 28, 27, 162, 216, 147, 58, 155, 153, 86, 317, 370, 233, 303, 189, 347, 79, 390, 390, 250, 239, 243, 249, 219, 99, 127, 101, 349, 376, 91, 196, 371, 159, 116, 273, 300, 226, 122, 62, 58, 232, 288, 329, 26, 55, 201, 211, 56, 19, 273, 318, 8, 160, 100, 198, 346, 101, 8, 252, 84, 216, 210, 242, 2, 354, 214, 43, 135, 18, 190, 338, 281, 329, 190, 32, 20, 394, 267, 117, 56, 198, 161, 191, 216, 352, 58, 136, 206, 66, 119, 225, 39, 103, 163, 234, 295, 150, 25, 100, 248, 393, 276, 313, 159, 286, 185, 272, 360, 328, 373, 279, 154, 230, 309, 377]
x = feature[0:-1]
y = feature[-1:]
x = np.reshape(x,newshape=(1,len(x)))
input_x = tf.keras.Input(shape=(None,))
embedding_x = layers.Embedding(num_classes, 1)(input_x)
hidden1 = layers.Dense(50,activation='relu')(embedding_x)
output = layers.Dense(400,activation='softmax')(hidden1)
model = tf.keras.Model(inputs = input_x,outputs = output)
embed_weight = model.get_layer(index = 1).weights
model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
              loss='mse',
              metrics=['accuracy'])
grads = model.optimizer.get_gradients(model.total_loss, embed_weight)
symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
f = KK.function(symb_inputs, grads)
x, y_, sample_weights = model._standardize_user_data(x, y)
output_grad = f(x + y_)
grad = output_grad[0].values
grad = np.reshape(np.array(grad),newshape=(1,len(grad)))
def update_embed_parameters(model, embed_grad):
    opt = model.optimizer
    embed_weights = []
    embed_layer = model.get_layer(index = 1)
    embed_weights.append(embed_layer.weights)
    grads = []
    now = 0
    for weight in embed_layer.weights:
        print(weight.name, weight.shape)
        sum = 1
        for i in range(10):
            try:
                sum = sum * weight.shape[i]
            except:
                break
        x = np.reshape(embed_grad[now:now + sum], newshape=weight.shape)
        grads.append(x)
        now = now + sum
        opt.apply_gradients(zip(grads, embed_layer.weights))
    return
update_embed_parameters(model,grad)
embed_layer = model.get_layer(index = 1)
embed_weights = embed_layer.weights
