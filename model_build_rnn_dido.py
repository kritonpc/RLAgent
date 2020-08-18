#!/usr/bin/env python
import keras, json;
import numpy as np;
	
def visionBranch(input_image):
	lstm = keras.layers.ConvLSTM2D(24, 5, strides=2, input_shape=(None, 30, 40, 3), data_format="channels_last", activation="relu")(input_image);
	conv = keras.layers.Conv2D(24, 3, strides=2, data_format="channels_last", activation="relu")(lstm);
	# conv = keras.layers.Conv2D(24, 3, strides=1, data_format="channels_last", activation="relu")(conv);
	return keras.layers.Flatten()(conv);
	# return keras.layers.Dense(96, activation="relu")(flat);

def createModel(lr):
	input_image = keras.Input(shape=(None, 30, 40, 3), name="lr_{}".format(lr));
	input_data = keras.Input(shape=(None, 3), name="data_input");
	
	vision = visionBranch(input_image);
	vision_dense = keras.layers.Dense(144, activation="relu")(vision);
	
	# steering_dense = keras.layers.Dense(96, activation="relu")(vision_dense);
	steering = keras.layers.Dense(13, activation="linear")(vision_dense);
	
	throttle_lstm = keras.layers.LSTM(12, input_shape=(None, 3), activation="relu")(input_data);
	throttle_dense = keras.layers.Dense(96, activation="relu")(vision_dense);
	throttle_conc = keras.layers.concatenate([throttle_dense, throttle_lstm]);
	# batch_norm = keras.layers.BatchNormalization(axis=1)(throttle_conc);
	throttle = keras.layers.Dense(9, activation="linear")(throttle_conc);
	
	model = keras.Model(inputs=[input_image, input_data], outputs=[steering, throttle])
	
	return model;

if __name__ == "__main__":
	model_name = input("Model name -> ");
	lr = float(input("Learning rate -> "));

	model = createModel(lr);
	if not model: raise Exception("Invalid model");

	model_dir = "models/" + model_name + "/";
	import os;
	if not os.path.exists(model_dir):
		os.makedirs(model_dir);

	keras.utils.plot_model(model, show_shapes=True, to_file=model_dir + "model.png");
	model.compile(loss=["mse", "mse"], optimizer=keras.optimizers.RMSprop(learning_rate=lr));
	model.save(model_dir + "model.h5");
	
	model.summary();
