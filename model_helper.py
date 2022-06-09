import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
import pickle
import os



class ChampModel():
	def __init__(self, model):
		self.model = model
		self.champ_names = pickle.load(open("lolViz/data/champ_names.pkl", "rb"))

	def champ_to_one_hot(self, champ_name):
		zero_one_hot = np.zeros(len(self.champ_names) + 1)
		zero_one_hot[self.champ_names.index(champ_name)] = 1
		return zero_one_hot

	def one_hot_to_champ(self, one_hot):
		return self.champ_names[np.argmax(one_hot)]

	def predict_champions(self, lol_hud):
		team_champs = {}
		for ii in range(2):
			champs = np.array([cv2.resize(champ_img, (25,25)) for champ_img in lol_hud.get_champ_imgs(ii)])
			preds = [(self.one_hot_to_champ(pred), max(pred)) for pred in self.model.predict(champs)]
			team_champs[ii] = preds
		return team_champs


class ClockModel():
	def __init__(self, model):
		self.model = model

	def _process_clock(self, clock):
		processed_nums = []
		thresh = cv2.threshold(clock , 70, 255, 0)[1]
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		numbers = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > 10]
		sorted_nums = sorted(numbers, key=lambda x: x[0])

		for number in sorted_nums:
			padded = np.zeros((12,12))
			x,y,w,h = number
			padded[:h , :w] = clock[y:y+h, x:x+w]
			processed_nums.append(padded)

		return processed_nums


	def predict_time(self, lol_img):
		nums = np.expand_dims(self._process_clock(lol_img.clock), -1)
		predictions = [list(pred).index(pred.max()) for pred in self.model.predict(nums)]
		seconds = ''.join([str(pred) for pred in predictions[-2:]])
		minutes = ''.join([str(pred) for pred in predictions[:-2]])
		return minutes, seconds








