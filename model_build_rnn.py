#!/usr/bin/env python
import keras, json;
import numpy as np;

# NN
def CreateModel(layers, learning_rate):
	if type(layers) != list: return None;
	
	model = keras.models.Sequential();

	# model.add(keras.layers.Conv2D(12, 5, strides=3, input_shape=(30, 40, 3), data_format="channels_last", activation="relu", name="lr"+str(learning_rate)));
	
	model.add(keras.layers.ConvLSTM2D(24, 5, strides=2, input_shape=(None, 30, 40, 3), data_format="channels_last", activation="relu", name="lr"+str(learning_rate)));
	
	#model.add(keras.layers.MaxPooling2D());

	model.add(keras.layers.Conv2D(24, 3, strides=2, data_format="channels_last", activation="relu"));
	# model.add(keras.layers.Conv2D(24, 3, data_format="channels_last", activation="relu"));
	# model.add(keras.layers.Conv2D(12, 3, data_format="channels_last", activation="relu"));
	#model.add(keras.layers.Conv2D(32, 5, data_format="channels_last", activation="relu"));
	#model.add(keras.layers.Conv2D(64, 3, data_format="channels_last", activation="relu"));
	#model.add(keras.layers.MaxPooling2D());

	#model.add(keras.layers.Conv2D(32, 3, data_format="channels_last", activation="relu"));

	#model.add(keras.layers.Conv2D(3, 3, data_format="channels_last", activation="tanh"));
	#model.add(keras.layers.Conv2D(16, 3, data_format="channels_last", activation="tanh"));
	#model.add(keras.layers.Conv2D(24, 5, data_format="channels_last", activation="tanh"));
	#model.add(keras.layers.AveragePooling2D());

	model.add(keras.layers.Flatten());
	#model.add(keras.layers.Dense(128, activation="relu"));
	#model.add(keras.layers.Dense(128, activation="relu"));
	#model.add(keras.layers.Dense(64, activation="relu"));
	#model.add(keras.layers.Dense(32, activation="relu"));
	model.add(keras.layers.Dense(96, activation="relu")); # pred spanim: bolo 64
	model.add(keras.layers.Dense(13, activation="linear")); # pred spanim: bolo 9

	#model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=["accuracy"]);
	model.compile(loss="mse", optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate));
	return model;


if __name__ == "__main__":
	model_name = input("Model name -> ");
	#layers = json.loads(input("Neurons in hidden layers as list (e.g. [6, 3, 3]) -> "));
	lr = float(input("Learning rate -> "));

	model = CreateModel([], lr);
	if not model: raise Exception("Invalid model");

	model_dir = "models/" + model_name + "/";
	import os;
	if not os.path.exists(model_dir):
		os.makedirs(model_dir);

	model.save(model_dir + "model.h5");
	keras.utils.plot_model(model, show_shapes=True, to_file=model_dir + "model.png");
	
	model.summary();
