import gym, gym_donkeycar;
import time;

original_reset = getattr(gym_donkeycar.envs.donkey_sim.DonkeyUnitySimHandler, "reset");
original_observe = getattr(gym_donkeycar.envs.donkey_sim.DonkeyUnitySimHandler, "observe");

def reset_new(self):
	self.lap_finished = False;
	self.lap_time = 0.0;
	self.line_crossed = None;
	self.frames = 0;
	
	original_reset(self);

def on_cross_start_new(self, data):
	if self.line_crossed is None:
		self.line_crossed = time.time();
		# print("[CROSS START NOT CROSSED]");
		return;
	
	if self.lap_finished: # wait for observe to notify
		# print("[CROSS START LAP FINISHED WAITING]");
		return;
	
	self.lap_finished = True;
	self.lap_time = round(time.time()-self.line_crossed, 2);
	self.line_crossed = time.time();
	self.frames = 2;
	# print("[CROSS START]", self.lap_finished, self.lap_time, self.line_crossed);

def observe_new(self):
	observation, reward, done, info = original_observe(self);
	
	info["lap_finished"] = self.lap_finished;
	info["lap_time"] = self.lap_time;
	
	if self.lap_finished:
		if self.frames > 0:
			self.frames -= 1;
		
		if self.frames <= 0:
			self.lap_finished = False;
			self.lap_time = 0.0;
		# print("[OBSERVE LAP FINISHED]", info["lap_finished"], info["lap_time"]);
	
	return observation, reward, done, info;

setattr(gym_donkeycar.envs.donkey_sim.DonkeyUnitySimHandler, "on_cross_start", on_cross_start_new);
setattr(gym_donkeycar.envs.donkey_sim.DonkeyUnitySimHandler, "reset", reset_new);
setattr(gym_donkeycar.envs.donkey_sim.DonkeyUnitySimHandler, "observe", observe_new);