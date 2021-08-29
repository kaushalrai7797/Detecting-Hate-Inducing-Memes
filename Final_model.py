# CNN Channel
#Input image of size (300,300,3)
main_input1 = Input(shape=(300,300,3))

# Convolution2D layers of size 64 
cnnlayer1 = Conv2D(64, kernel_size=(5, 5),activation='relu')(main_input1)
pool_layer1=MaxPooling2D(pool_size=(5, 5))(cnnlayer1)

#Droput layer to prevent overfitting

drop1=Dropout(0.4)(pool_layer1)

# Convolution2D layers of size 32
cnnlayer2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(drop1)
pool_layer2=MaxPooling2D(pool_size=(3, 3))(cnnlayer2)

# Flatten layer to covert 3D feature map to 1D
flat=Flatten()(pool_layer2)
l1=Dense(32, activation='relu')(flat)


#LSTM channel
#Input text from memes to LSTM channel
main_input2 = Input(shape=(sequence_length,))

# EMBEDDING LAYER - DISTRIBUTED REPRESENTATION OF TWEETS 
# EMBEDDINGS - GLOVE,Twiiter Word2vec,Fasttext,Bert 100 dimensions further trained on davidson and heot dataset after proper preprocessing
# EMBEDDING DIMENSION = 100
embed2 = Embedding(len(vocab)+1, EMBEDDING_DIMENSION, weights=[embedding], name='embedding_layer2')(main_input2)

#Droput layer to prevent overfitting
drop3=Dropout(0.2)(embed2)

# LSTM of size 64
lstmmodel = LSTM(64,dropout_W=0.4,dropout_U=0.4)(drop3)

# Series of Dense layers
dl1 = Dense(64, activation='relu')(lstmmodel)
l2 = Dense(32, activation='relu')(dl1)


# Recombination of channels
# Concatening outputs from both channels
model = concatenate([l1, l2])
d = Dense(32, activation='relu')(model)
output = Dense(3, activation='softmax',name='last')(d)
model=Model(inputs=[main_input1, main_input2], outputs=output)

# Compining model using adam optimizer and categorical crossentropy loss
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
