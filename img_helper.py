import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import measure
from numpy.lib.stride_tricks import as_strided


class LoLImage():
	def __init__(self, image_arr, clock_coords = (220, 35, 50, 20)):
		self.image_arr = image_arr
		self.clock = self._grab_clock(clock_coords)
		self.y = image_arr.shape[0]
		self.x = image_arr.shape[1]
		self.minimap = self._separate_minimap()
		self.hud = self._separate_hud()
		self.left_team, self.right_team = self._separate_teams()
		self.left_hpbars, self.right_hpbars = self._separate_hpbars()


	def _grab_clock(self, clock_coords):
		x,y,w,h = clock_coords
		return cv2.cvtColor(self.image_arr[y:y+h, x:x+w], cv2.COLOR_RGB2GRAY)


	def _separate_hpbars(self):
		left_hpbars = [self.left_team[ii+30:ii+42, 20:48] for ii in range(31,self.left_team.shape[0],69)]
		right_hpbars = [self.right_team[ii+30:ii+42, 30:58] for ii in range(31,self.right_team.shape[0],69)]
		return left_hpbars, right_hpbars

	def _separate_teams(self):
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

	def _separate_minimap(self):
		mini_b = {
			'x':(round(0.855*self.x), round(0.996*self.x)),
			'y':(round(0.736*self.y), round(0.986*self.y))
		}
		minimap = self.image_arr.copy()[mini_b['y'][0]:mini_b['y'][1], mini_b['x'][0]:mini_b['x'][1]]

		return minimap

	def _separate_hud(self):
		hud_b = {
			'x':(round(0.315*self.x), round(0.69*self.x)),
			'y':(round(0.785*self.y), 0)
		}

		hud = self.image_arr.copy()[hud_b['y'][0]:, hud_b['x'][0]:hud_b['x'][1]]

		return hud

	def _predict_hp_mana(self, img):
		if len(np.where(img[:,:,1].ravel() > 175)[0]) == 0:
			return 0, 0, False
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		__, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		two_contours = sorted(contours, key=lambda x: cv2.contourArea(x))[-2:]
		sorted_contours = sorted(two_contours, key=lambda x: cv2.boundingRect(x)[1])
		try:
			__, __, hp_w, __ = cv2.boundingRect(sorted_contours[0])
		except:
			hp_w = -27
		
		try:
			__, __, mana_w, __ = cv2.boundingRect(sorted_contours[1])
		except:
			mana_w = -27

		hp = hp_w / 27
		mana = mana_w / 27
		return round(hp,2), round(mana,2), True

	def _put_text(self, img, text, pos, color=(255,255,255)):
		fontFace = cv2.FONT_HERSHEY_TRIPLEX
		fontScale = 0.85
		color = color
		thickness = 2
		cv2.putText(img, text, pos, fontFace=2, fontScale=fontScale, color=color, thickness=thickness)


	def predict_team_hp_mana(self, team_side):
		if type(team_side) != int:
			raise Exception("Pass in 0 - left, or 1 - right")

		if team_side == 0:
			return [self._predict_hp_mana(hpbar) for hpbar in self.left_hpbars]
		elif team_side == 1:
			return [self._predict_hp_mana(hpbar) for hpbar in self.right_hpbars]

	def display_team_hpbars(self):
		fig, axs = plt.subplots(5, 2, figsize=(15,15))
		for ii, hpbar in enumerate(self.left_hpbars):
			axs[ii, 0].imshow(hpbar)
		for ii, hpbar in enumerate(self.right_hpbars):
			axs[ii, 1].imshow(hpbar)
		return plt

	def label_hp_mana(self):
		left_data = self.predict_team_hp_mana(0)
		right_data = self.predict_team_hp_mana(1)
		a = self.image_arr.copy()
		hp = 0.5
		mana = 1.0
		alive=True
		self._put_text(a, "left team", (75, 80))
		for jj, data in enumerate(left_data):
			hp, mana, alive = round(data[0], 2), round(data[1], 2), data[2]

			id_y = 120+95*jj
			self._put_text(a, f"id: {jj}", (200,id_y))
			if alive == True:
				self._put_text(a, f"hp: {hp}", (240,id_y+30), color=(0,255,0))
				self._put_text(a, f"mana: {mana}", (260,id_y+60), color=(20,20,255))
			else:
				self._put_text(a, f"DEAD", (220,id_y+45), color=(255,0,0))

		self._put_text(a, "right team", (1200-150, 80))
		for jj, data in enumerate(right_data):
			hp, mana, alive = round(data[0], 2), round(data[1], 2), data[2]

			id_y = 120+95*jj
			self._put_text(a, f"id: {jj}", (1200-200,id_y))
			if alive == True:
				self._put_text(a, f"hp: {hp}", (1200-260,id_y+30), color=(0,255,0))
				self._put_text(a, f"mana: {mana}", (1200-320,id_y+60), color=(20,20,255))
			else:
				self._put_text(a, f"DEAD", (1200-220,id_y+45), color=(255,0,0))

		return a
				

	def plot_all(self):
		fig, axs = plt.subplots(2,2, figsize=(20,20))
		axs[0,0].imshow(self.minimap)
		axs[0,1].imshow(self.hud)
		axs[1,0].imshow(self.left_team)
		axs[1,1].imshow(self.right_team)
		return plt


class LoLHud():
	def __init__(self, hud_image, predict_cs = False, cs_model = None):
		self.image_arr = hud_image
		self.gray = cv2.cvtColor(hud_image, cv2.COLOR_BGR2GRAY)
		self.y = hud_image.shape[0]
		self.x = hud_image.shape[1]
		self.cs_model = cs_model
		self.left_team, self.right_team = self.split_teams()
		self._chop_team_imgs()
		if predict_cs:
			self.build_cs_dfs()

	def split_teams(self):
		left_team = self.image_arr[:, :(self.x//2)]
		right_team = self.image_arr[:, (self.x//2):]
		return left_team, right_team

	def _chop_team_imgs(self):
		team_x = self.left_team.shape[1]

		self.left_wards = self.left_team[:, :round(0.12*team_x)]
		self.left_items = self.left_team[:, round(0.12*team_x):round(0.554*team_x)]
		self.left_kda = self.left_team[:, round(0.554*team_x):round(0.76*team_x)]
		self.left_cs = self.image_arr[:, round(0.38*self.x):round(0.44*self.x)]
		self.left_champ_imgs = self.left_team[:, round(0.88*team_x):]

		self.right_wards = self.right_team[:, round(0.917*team_x):]
		self.right_items = self.right_team[:, round(0.458*team_x):round(0.917*team_x)]
		self.right_kda = self.right_team[:, round(0.27*team_x):round(0.458*team_x)]
		self.right_cs = self.image_arr[:, round(0.575*self.x):round(0.635*self.x)]
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
		predictions = self.cs_model.predict(input_ready_array, 1, verbose=0)[0]
		predicted_num = list(predictions).index(max(predictions))
		return predicted_num

	def _cs_df(self, cs_img):
		if self.cs_model == None:
			raise Exception("Set the LoLHud model to a working MNIST model or pass one in")

		#Generate contours
		gray = cv2.cvtColor(cs_img, cv2.COLOR_BGR2GRAY)
		thresh = self._threshold(gray, 65)
		contours = self._contour_thresholded_image(thresh)
		if len(contours) == 0:
			raise Exception("Was unable to find any contours")

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

	def get_champ_imgs(self, team):
		if type(team) is not int:
			raise Exception("Enter 0 or 1 as team")
		if team == 0:
			champ_imgs = self.left_champ_imgs
		elif team == 1:
			champ_imgs = self.right_champ_imgs
		gray = cv2.cvtColor(champ_imgs, cv2.COLOR_RGB2GRAY)
		thresh = cv2.threshold(gray, 30, 255, 0)[1]
		contours, __ = cv2.findContours(thresh, 0, 2)
		contours = sorted(contours, key=lambda x: cv2.contourArea(x))[len(contours) - 5:] #Get top 5 biggest contours
		bounding_rects = [cv2.boundingRect(contour) for contour in contours] #Get bounding rects
		bounding_rects = sorted(bounding_rects, key=lambda x: x[1]) #Sort from top to bottom
		imgs = [champ_imgs[y:y+h, x:x+w] for x,y,w,h in bounding_rects]
		return imgs
		
	def build_cs_dfs(self):
		self.left_cs_df = self._cs_df(self.left_cs)
		self.right_cs_df = self._cs_df(self.right_cs)


class LoLMinimap():
	def __init__(self, image_arr, base_img=None):
		self.image_arr = image_arr
		self.median_blur_kernel = np.array([[1/4,1/4],
											[1/4,1/4]])
		self.base_img = base_img

	def _strided_convolution(self, image, weight, stride):
		im_h, im_w = image.shape
		f_h, f_w = weight.shape

		out_shape = (1 + (im_h - f_h) // stride, 1 + (im_w - f_w) // stride, f_h, f_w)
		out_strides = (image.strides[0] * stride, image.strides[1] * stride, image.strides[0], image.strides[1])
		windows = as_strided(image, shape=out_shape, strides=out_strides)

		return np.tensordot(windows, weight, axes=((2, 3), (0, 1)))

	def _process(self, img):
		blurred = self._strided_convolution(img, self.median_blur_kernel,1).astype(np.float32)
		pooled = measure.block_reduce(blurred, (2,2), np.max).astype(np.float32)
		return pooled

	def _processed_template_matching(self, img, template):
		combined_res = np.zeros((84,84))
		for ii in range(3):
			c_mm = img[:,:,ii]
			c_champ = template[:,:,ii]
			processed_mm = self._process(c_mm)
			processed_champ = self._process(c_champ)
			combined_res+=cv2.matchTemplate(processed_mm, processed_champ, eval('cv2.TM_SQDIFF_NORMED'))
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(combined_res)
		pos = np.array(min_loc)*2
		return pos

	def _narrow_search_space(self, thresh_val = 15):
		if self.base_img is None:
			return self.image_arr

		(r,g,b) = (np.array(channel, dtype=np.float32) for channel in cv2.split(self.image_arr))
		(r_base,g_base,b_base) = (np.array(channel, dtype=np.float32) for channel in cv2.split(self.base_img))
		(new_r, new_g, new_b) = (np.abs(main - base) for main, base in zip([r,g,b], [r_base,g_base,b_base]))
		new_image = np.dstack([new_r, new_g, new_b]).astype(np.uint8)

		thresh = cv2.threshold(cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY), thresh_val, 255, 0)[1]
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		contours = [contour  for contour in contours if cv2.contourArea(contour) > 250]
		bw_image = cv2.drawContours(np.zeros(self.image_arr.shape), contours, -1, (255,255,255), -1) / 255
		altered_mm = self.image_arr.copy()
		for channel in range(3):
			for ii in range(180):
				for jj in range(181):
					altered_mm[ii,jj,channel] = altered_mm[ii,jj,channel] * bw_image[ii,jj,channel]

		return altered_mm

	def predict_champion_positions(self, champ_icons):
		if type(champ_icons) != dict:
			raise Exception("Pass in a dictionary {champion_name:champion_icon} with the champion icon scaled to (15,15)")
		champ_pos = {str(champ): self._processed_template_matching(self.image_arr, champ_img)
					for champ, champ_img in champ_icons.items()}
		return champ_pos

	def draw_labels(self, champ_dict):
		scale_factor = 5
		drawn_mm = cv2.resize(self.image_arr.copy(), (0,0), fx=scale_factor, fy=scale_factor)
		for champ, pos in champ_dict.items():
			cv2.rectangle(drawn_mm, pos*scale_factor,(pos*scale_factor)+np.array([scale_factor*15,scale_factor*15]), (0,255,0), 3)
			cv2.putText(drawn_mm, str(champ), pos*scale_factor, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=1)
		return drawn_mm


