import keras, random, json;
import numpy as np;

# double output agent
class DQNAgentDIDO(object):
	def __init__(self, model_location, epsilon=1.0):
		self.model_location = model_location;
		self.model = keras.models.load_model(model_location);
		self.target_model = keras.models.load_model(model_location);
		self.action_space = [self.model.layers[-(i+1)].output_shape[1] for i in range(2)];
		self.action_space.reverse();
		self.memory = [];
		self.replays = 0;
		self.max_memory = 20000;

		self.gamma = 0.99;							# long term value
		self.epsilon = max(epsilon, 0.0);			# exploration rate
		self.epsilon = min(self.epsilon, 1.0);
		self.epsilon_min = 0.05;
		self.epsilon_decay = 0.99935;

		self.lastrandom = False;
		self.lastrandoms = [];

		# shape = self.model.layers[i].input_shape[0]
		# shape = [x if x != None else 1 for x in shape]
		# tmp = np.zeros(tuple(shape));
		# self.model.predict_on_batch(tmp);
		# self.target_model.predict_on_batch(tmp);

		# if epsilon > 0.0:
			# tmptargets = [np.zeros((1, self.action_space[i])) for i in range(2)];
			# self.model.fit(tmp, tmptargets, verbose=0)
			# self.target_model.fit(tmp, tmptargets, verbose=0)

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done));

		if len(self.memory) > self.max_memory: self.memory = self.memory[-self.max_memory:];

	def merge_models(self):
		self.target_model.set_weights(self.model.get_weights());
		
	def act(self, state):
		act_values = self.model.predict_on_batch(state);
		actions = [];
		Qs = [];
		
		for i in range(2):
			if random.random() <= self.epsilon:
				actions.append(random.randint(0, self.action_space[i]-1))
				Qs.append(None);
			else:
				actions.append(np.argmax(act_values[i][0]))
				Qs.append(np.amax(act_values[i][0]))
		
		return actions, Qs
		

	def replay(self, batch_size):
		# select random samples from memory
		minibatch = random.sample(self.memory, batch_size);
		
		images = []
		data = []
		actions = []
		rewards = []
		next_images = []
		next_data = []
		dones = []
		for state, action, reward, next_state, done in minibatch:
			images.append(state[0])
			data.append(state[1])
			actions.append(action)
			rewards.append(reward)
			next_images.append(next_state[0]);
			next_data.append(next_state[1]);
			dones.append(done)
		
		images = np.array(images, dtype=np.float16)
		data = np.array(data, dtype=np.float16)
		next_images = np.array(next_images, dtype=np.float16)
		next_data = np.array(next_data, dtype=np.float16)

		current_r = self.model.predict_on_batch([images, data]);
		target_r = self.target_model.predict_on_batch([next_images, next_data]);
		targetn_r = [[np.amax(x) for x in target_r[i]] for i in range(2)];
		ytrain1 = [];
		ytrain2 = []

		for (i, reward) in enumerate(rewards):
			current_r[0][i][actions[i][0]] = reward + ((targetn_r[0][i]*self.gamma) if not dones[i] else 0.0); # steering
			current_r[1][i][actions[i][1]] = reward + ((targetn_r[1][i]*self.gamma) if not dones[i] else 0.0); # throttle
			
			ytrain1.append(current_r[0][i]);
			ytrain2.append(current_r[1][i]);

		xtrain = [np.array(images, dtype=np.float16), np.array(data, dtype=np.float16)]
		ytrain = [np.array(ytrain1, dtype=np.float16), np.array(ytrain2, dtype=np.float16)]

		self.model.train_on_batch(xtrain, ytrain);
		self.replays += 1;

		# update epsilon to minimize exploration on the long term
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay;
		elif self.epsilon < self.epsilon_min:
			self.epsilon = self.epsilon_min;
	
	def save(self, model_name=None):
		try:
			location = self.model_location.replace("model_trained.h5", "model_trained.h5" if not model_name else model_name+".h5");
			location = location.replace("model.h5", "model_trained.h5" if not model_name else model_name+".h5");
			self.model.save(location);
			
		except Exception as e:
			print("Exception", str(e));
