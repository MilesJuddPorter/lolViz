import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class LoLImage():
	def __init__(self, image_arr):
		self.image_arr = image_arr
		self.y = image_arr.shape[0]
		self.x = image_arr.shape[1]
		self.minimap = self.separate_minimap()
		self.hud = self.separate_hud()
		self.left_team, self.right_team = self.separate_teams()

	def separate_teams(self):
		lb = {
			'x':(0, round(0.0586*self.x)),
			'y':(round(0.104*self.y), round(0.625*self.y))
		}

		rb = {
			'x':(round(0.9375*self.x), 0),
			'y':(round(0.104*self.y), round(0.625*self.y))
		}

		left_team = self.image_arr.copy()[lb['y'][0]:lb['y'][1], :lb['x'][1]]
		right_team = self.image_arr.copy()[rb['y'][0]:rb['y'][1], rb['x'][0]:]

		return left_team, right_team	

	def separate_minimap(self):
		mini_b = {
			'x':(round(0.855*self.x), round(0.996*self.x)),
			'y':(round(0.736*self.y), round(0.986*self.y))
		}
		minimap = self.image_arr.copy()[mini_b['y'][0]:mini_b['y'][1], mini_b['x'][0]:mini_b['x'][1]]

		return minimap

	def separate_hud(self):
		hud_b = {
			'x':(round(0.315*self.x), round(0.69*self.x)),
			'y':(round(0.785*self.y), 0)
		}

		hud = self.image_arr.copy()[hud_b['y'][0]:, hud_b['x'][0]:hud_b['x'][1]]

		return hud



	def plot_all(self):
		fig, axs = plt.subplots(2,2, figsize=(20,20))
		axs[0,0].imshow(self.minimap)
		axs[0,1].imshow(self.hud)
		axs[1,0].imshow(self.left_team)
		axs[1,1].imshow(self.right_team)
		return plt



class LoLHud():
	def __init__(self, hud_image, predict_cs = False, model = None):
		self.image_arr = hud_image
		self.gray = cv2.cvtColor(hud_image, cv2.COLOR_BGR2GRAY)
		self.y = hud_image.shape[0]
		self.x = hud_image.shape[1]
		#self.left_cs, self.right_cs = self._separate_cs()
		self.model = model
		self.left_team, self.right_team = self.split_teams()
		self._chop_team_imgs()
		if predict_cs:
			self.build_cs_dfs()

	def split_teams(self):
		left_team = self.img[:, :(self.x//2)]
		right_team = self.img[:, (self.x//2):]
		return left_team, right_team

	# def _separate_cs(self):
	# 	left_cs = self.image_arr.copy()[:, round(0.38*self.x):round(0.44*self.x)]
	# 	right_cs = self.image_arr.copy()[:, round(0.575*self.x):round(0.635*self.x)]

	# 	return left_cs, right_cs

	def _chop_team_imgs(self):
		team_x = self.left_team.shape[1]

		self.left_wards = self.left_team[:, :round(0.12*team_x)]
		self.left_items = self.left_team[:, round(0.12*team_x):round(0.554*team_x)]
		self.left_kda = self.left_team[:, round(0.554*team_x):round(0.76*team_x)]
		self.left_cs = self.left_team[:, round(0.76*team_x):round(0.88*team_x)]
		self.left_champ_imgs = self.left_team[:, round(0.88*team_x):]

		self.right_wards = self.right_team[:, round(0.917*team_x):]
		self.right_items = self.right_team[:, round(0.458*team_x):round(0.917*team_x)]
		self.right_kda = self.right_team[:, round(0.27*team_x):round(0.458*team_x)]
		self.right_cs = self.right_team[:, :round(0.146*team_x):round(0.27*team_x)]
		self.right_champ_imgs = self.right_team[:, :round(0.146*team_x)]

	def _threshold(self, gray, value):
		__, thresh = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)
		return thresh

	def _contour_thresholded_image(self, thresh):
		edged = cv2.Canny(thresh, 30, 200)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		return contours
	
	def _predict_on_contour(self, contour, thresh):

		#Prep image by upscaling contour and placing it on 28x28 padded array
		x,y,w,h = cv2.boundingRect(contour)
		trimmed_array = thresh[y:y+h , x:x+w]
		upscaled_array = cv2.resize(trimmed_array, (0,0), fx=1.5, fy=1.5)
		padded_array = np.zeros((28, 28))
		shape = upscaled_array.shape
		#try except to correct for even/odd
		top_left = (14-shape[0]//2, 14-shape[1]//2)
		padded_array[top_left[0]:top_left[0] + shape[0], top_left[1]:top_left[1] + shape[1]] = upscaled_array
		input_ready_array = padded_array.reshape(1,28,28,1)

		#Predict on prepped array
		predictions = self.model.predict(input_ready_array, 1, verbose=0)[0]
		predicted_num = list(predictions).index(max(predictions))
		return predicted_num

	def _cs_df(self, cs_img):
		if self.model == None:
			raise Exception("Set the LoLHud model to a working MNIST model or pass one in")

		#Generate contours
		gray = cv2.cvtColor(cs_img, cv2.COLOR_BGR2GRAY)
		thresh = self._threshold(gray, 65)
		contours = self._contour_thresholded_image(thresh)

		#Create the dataframe for each contour
		numbers_found = []
		for contour in contours:
			x,y,w,h = cv2.boundingRect(contour)
			try:
				prediction = self._predict_on_contour(contour, thresh)
			except:
				prediction = -1
			data = {'num_found': prediction,
					'x': x,
					'y': y}
			numbers_found.append(data)
		num_df = pd.DataFrame(numbers_found)

		#Cluster the contour dataframe to bring rows together (1, 3, 5 turns into 135 on one row)
		km = KMeans(n_clusters = 5).fit(num_df[['x','y']])
		num_df['labels'] = km.labels_
		cs_df = num_df.groupby('labels').min()
		for label in num_df.labels.unique():
			value = ''.join(num_df[num_df.labels == label].sort_values('x')['num_found'].astype('str').values)
			if '-1' in value:
				cs_df.iloc[label,0] = 'error'
			else:
				cs_df.iloc[label,0] = value
		cs_df.sort_values('y', inplace=True)
		return cs_df

	def build_cs_dfs(self):
		self.left_cs_df = self._cs_df(self.left_cs)
		self.right_cs_df = self._cs_df(self.right_cs)