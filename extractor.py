import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from lolViz.img_helper import *
from lolViz.video_helper import *
from lolViz.model_helper import *



class Game():
	def __init__(self, clock_coords = (610, 50, 60, 20)):
		#Macro
		self.team1_name = None
		self.team2_name = None
		self.clock_coords = clock_coords


		#Team champions
		self.left_team_champs = {ii: None for ii in range(5)}
		self.right_team_champs = {ii: None for ii in range(5)}
		self.champion_predictions = {ii: np.zeros(181) for ii in range(10)}

		#Models
		self.champ_model = ChampModel(load_model("lolViz/models/champ_detector"))
		self.in_game_model = load_model("lolViz/models/in_game_v1")
		self.clock_model = ClockModel(load_model("lolViz/models/clock_cnn"))
		self.cs_model = load_model("lolViz/models/mnist_LeNet.h5")


		self.frames_analyzed = 1

		#Data
		self.data = {"champs":{"left_team": self.left_team_champs,"right_team": self.right_team_champs},
					 "Frames": []}

	def analyze_image(self, img):
		#Check if the current image is a league game
		smol_img = cv2.resize(img, (640, 360))
		in_game_pred = self.in_game_model.predict(np.expand_dims(smol_img, 0))
		if in_game_pred[0] < 0.9: 
			return None

		lol_img = LoLImage(img, clock_coords=self.clock_coords)
		minutes, seconds = self.clock_model.predict_time(lol_img)

		#Increase frames analyzed
		self.frames_analyzed += 1

		#Update champion predictions
		if self.frames_analyzed < 20:
			self.update_champion_predictions(lol_img)

		self.add_frame(lol_img)


	def update_champion_predictions(self, lol_img):
		lol_hud = LoLHud(lol_img.hud)
		champ_preds = self.champ_model.predict_champions(lol_hud)
		if len(champ_preds[0]) + len(champ_preds[1]) == 10:
			for ii in range(2):
				for jj, (champ_name, confidence) in enumerate(champ_preds[ii]):
					ohe = self.champ_model.champ_to_one_hot(champ_name) * confidence
					self.champion_predictions[(ii*5)+jj] += (ohe/self.frames_analyzed)

		for ii in range(5):
			t1_pred, t2_pred = self.champion_predictions[ii], self.champion_predictions[ii+5]
			self.left_team_champs[ii] = self.champ_model.one_hot_to_champ(t1_pred)
			self.right_team_champs[ii] = self.champ_model.one_hot_to_champ(t2_pred)

	def pred_time(self, lol_img):
		minutes, seconds = self.clock_model.predict_time(lol_img)
		return minutes, seconds

	def get_hp_mana(self, img):
		lol_img = LoLImage(img, clock_coords=self.clock_coords)
		
		return lol_img.predict_team_hp_mana(0)

	def add_frame(self, lol_img):
		lol_hud = LoLHud(lol_img.hud, predict_cs=True, cs_model=self.cs_model)
		lol_minimap = LoLMinimap(lol_img.minimap)

		minutes, seconds = self.clock_model.predict_time(lol_img)

		team1_hp_mana = lol_img.predict_team_hp_mana(0)
		team2_hp_mana = lol_img.predict_team_hp_mana(1)

		data = {"team1":{},
				"team2":{}}
		for ii in range(5):
			data["team1"][ii] = {item: value for item, value in zip(["hp", "mana", "alive"],
				team1_hp_mana[ii])}
			data["team2"][ii] = {item: value for item, value in zip(["hp", "mana", "alive"],
				team2_hp_mana[ii])}
			data["team1"][ii]["cs"] = list(lol_hud.left_cs_df['num_found'])[ii]
			data["team2"][ii]["cs"] = list(lol_hud.right_cs_df['num_found'])[ii]

		self.data["Frames"].append({
			"time": {"minutes": minutes, "seconds":seconds},
			"team1": data["team1"],
			"team2": data["team2"]
			})


