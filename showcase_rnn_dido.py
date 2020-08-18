import json, time
import numpy as np
import gym, gym_donkeycar, donkeycar_modified
from agent_dido import DQNAgentDIDO
import tensorflow as tf
import cv2, random
from utils import *;
	
def main():
	# enable GPU memory growth
	physical_devices = tf.config.list_physical_devices('GPU') 
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
	
	# model
	model_name = input("Model name -> ");
	model_file = input("Model file -> ");
	my_model = "models/{}/{}.h5".format(model_name, model_file);
	
	epsilon = float(input("Epsilon -> "));
	episode_count = int(input("Episode count -> "));
	
	print("Loading", my_model, "with epsilon", epsilon);
	agent = DQNAgentDIDO(my_model, float(epsilon));
	
	# information
	resizeScale = (40, 30)
	frame_n = 3;
	max_cte = 4.35;
	
	# statistics
	score = [];
	rewards = [];
	highest_score = 0;
	highest_reward = 0;
	max_score = None;
	
	# velocity
	max_velocity = 10.0
	max_throttle = 0.5;
	throttle_step = 1.5*max_throttle/(agent.action_space[1]-1)
	throttle_table = [i*throttle_step-max_throttle/2.0 for i in range(agent.action_space[1])];
	
	# steering
	max_steering = 0.75
	steering_step = 2*max_steering/(agent.action_space[0]-1)
	steering_table = [i*steering_step-max_steering for i in range(agent.action_space[0])]
	
	# setup donkey environment
	conf = {
		# "exe_path":"remote",
		"exe_path":"D:/sdsandbox/build2/donkey_sim.exe",
		"host":"127.0.0.1",
		"port":9094,
		"body_style":"donkey",
		"body_rgb":(128, 128, 128),
		"car_name":"rl",
		"font_size":100
	}
	
	env = gym.make("donkey-generated-roads-v0", conf=conf)
	# env = gym.make("donkey-generated-track-v0", conf=conf)
	env.viewer.handler.max_cte = max_cte;
	cv2.namedWindow("camera");
	
	start = time.time();
	first_start = start;
	
	for e in range(episode_count):		
		# at each episode, reset environment to starting position
		image = env.reset();
		images = np.empty((frame_n, resizeScale[1], resizeScale[0], 3));
		data = np.empty((frame_n-1, 3));
		
		images[0] = preprocessImage(image, resizeScale);
		need_frames = frame_n-1;
		
		done = False;
		score.append(0);
		rewards.append(0.0);
		last_velocity = [0.0];
		laps = 0;
		start = time.time();
		
		while not done and (score[-1] < max_score if max_score else True):
			if need_frames > 0:
				action = [random.randint(0, agent.action_space[0]-1), random.randint(0, agent.action_space[1]-1)]
				steering = steering_table[action[0]];
				throttle = throttle_table[action[1]];
				
				next_image, reward, done, info = env.step([steering, throttle]);
				
				images[frame_n-need_frames] = preprocessImage(next_image, resizeScale);
				data[frame_n-need_frames-1] = np.array([info["speed"]/max_velocity, action[0]/(agent.action_space[0]-1), action[1]/(agent.action_space[1]-1)]);
				need_frames -= 1
				
				last_velocity.append(info["speed"]);
				continue
				
			# select action, observe environment, calculate reward
			action, Qs = agent.act([np.asarray([images]), np.asarray([data])]);
			steering = steering_table[action[0]];
			throttle = throttle_table[action[1]];

			next_image, reward, done, info = env.step([steering, throttle]);
			
			img = cv2.resize(next_image, (320, 240), interpolation=cv2.INTER_AREA)
			cv2.imshow("camera", img)
			
			last_velocity.append(round(info["speed"], 4));
			reward = 0.0;
			if abs(info["cte"]) >= max_cte:
				done = True
				reward = -3.0
			else:
				reward += (1.0 - (abs(info["cte"]) / max_cte)*2.0) * 0.33;
				reward += min((last_velocity[-1]/max_velocity)*2 - 1.0, 1.0) * 0.67;
			
			# for track
			# if info["lap_finished"]:
				# if info["lap_time"] >= 15.0:
					# laps += 1;
				# else: # we probably just turned around lol
					# print("[{}] LAP TOO QUICK?! {}s".format(name, info["lap_time"]));
					# break;
			
			score[-1] += 1;
			rewards[-1] += reward;
			
			next_images = np.roll(images, -1, axis=0);
			next_images[-1] = preprocessImage(next_image, resizeScale);
			
			next_data = np.roll(data, -1, axis=0);
			next_data[-1] = np.array([last_velocity[-1]/max_velocity, action[0]/(agent.action_space[0]-1), action[1]/(agent.action_space[1]-1)]);
			
			images = next_images;
			data = next_data;
			
			cv2.waitKey(1)
		
		env.step([0.0, -0.03]);
		
		if len(score) > 20: score = score[-20:];
		if len(rewards) > 20: rewards = rewards[-20:];
			
		if score[-1] >= highest_score:
			highest_score = score[-1];
		
		if rewards[-1] >= highest_reward:
			highest_reward = rewards[-1]
		
		print("episode: {}/{}, score: {}, reward: {}, laps: {}, e: {:.2}"
				.format(e+1, episode_count, score[-1], round(rewards[-1], 2), laps, round(agent.epsilon, 2)));
		
		if (e+1) % 5 == 0:
			print("Took", round((time.time()-start)/60, 2), "minutes\n");
			start = time.time();

	print("Showcase time:", round((time.time()-first_start)/60, 2), "minutes");

if __name__ == "__main__":
	main();