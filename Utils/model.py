import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Flatten, Dense, Embedding, Dropout, Concatenate
from tensorflow.keras.metrics import Mean, RootMeanSquaredError, MeanAbsoluteError

# Model Definition
class MLP(Layer):
    def __init__(self, first_node, n_layer):
        super(MLP, self).__init__()
        n_node = first_node
        self.mlp_layer = Sequential()
        for i in range(n_layer):
            self.mlp_layer.add(Dense(units = n_node, activation = "relu"))
            self.mlp_layer.add(Dropout(0.1))
            n_node //= 2

    def call(self, input):
        x = self.mlp_layer(input)
        return x
    
class MFNR(Model):
    def __init__(self, N, M, K):
        super(MFNR, self).__init__()
        self.user_embedding = Embedding(N, K)
        self.item_embedding = Embedding(M, K)
        self.user_flatten = Flatten()
        self.item_flatten = Flatten()
        self.user_nlp_concat = Concatenate()
        self.user_nlp_MLP = MLP(512, 4)
        self.item_nlp_concat = Concatenate()
        self.item_nlp_MLP = MLP(512, 4)
        self.user_concat = Concatenate()
        self.item_concat = Concatenate()
        self.rating_concat = Concatenate()
        self.rating_mlp = MLP(64, 3)
        self.output_layer =  Dense(1,activation = "linear")

    def call(self, inputs):
        # User, Item Embedding
        user_emb = self.user_embedding(inputs[0])
        item_emb = self.item_embedding(inputs[1])
        user_emb = self.user_flatten(user_emb)
        item_emb = self.item_flatten(item_emb)

        # User NLP & MLP
        user_nlp = self.user_nlp_concat([inputs[2], inputs[4]])
        user_nlp = self.user_nlp_MLP(user_nlp)
        # Item NLP
        item_nlp = self.item_nlp_concat([inputs[3], inputs[5]])
        item_nlp = self.item_nlp_MLP(item_nlp)

        # User Representation
        user_rep = self.user_concat([user_emb, user_nlp])
        # Item Representation
        item_rep = self.item_concat([item_emb, item_nlp])

        # Rating Prediction
        MLP = self.rating_concat([user_rep, item_rep])
        MLP = self.rating_mlp(MLP)
        output = self.output_layer(MLP)
        return output
    

# Model Evaluation
def load_metrics():
    global train_loss, train_acc
    global validation_loss, validation_acc
    global test_loss, test_rmse, test_mae

    train_loss = Mean()
    validation_loss = Mean()
    test_loss = Mean()

    test_rmse = RootMeanSquaredError()
    test_mae = MeanAbsoluteError()

@tf.function
def trainer():
    global train_tfds, train_loss, model
    global optimizer, loss_object

    for user, item, user_bert, item_bert, user_roberta, item_roberta, y in train_tfds:
        with tf.GradientTape() as tape:
            predictions = model([user, item, user_bert, item_bert, user_roberta, item_roberta])
            loss = loss_object(y, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)

@tf.function
def validation():
    global validation_tfds, model, loss_object
    global validation_loss, validation_acc
    for user, item, user_bert, item_bert, user_roberta, item_roberta, y in validation_tfds:
        predictions = model([user, item, user_bert, item_bert, user_roberta, item_roberta])
        loss = loss_object(y, predictions)

        validation_loss(loss)

@tf.function
def tester():
    global test_tfds, best_model, loss_object
    global test_loss, test_rmse, test_mae
    for user, item, user_bert, item_bert, user_roberta, item_roberta, y in test_tfds:
        predictions = best_model([user, item, user_bert, item_bert, user_roberta, item_roberta])
        loss = loss_object(y, predictions)

        test_loss(loss)
        test_rmse(y, predictions)
        test_mae(y, predictions)