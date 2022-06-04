import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import ffmpeg


class VideoHelper():
	def __init__(self, model = None):
		self.model = model


	def imgs_to_video(self, path_to_images, framerate, output_file, file_ending = ".jpg"):
		"""
		path_to_images: path to folder to load imgs from
		file_ending: .jpg default with option to pass other file endings
		framerate: framerate to save viceo as
		output_file: file to save (could also pass folder prefix)
		"""
		try:
			(
			ffmpeg
			.input(f'{path_to_dump}/%d.{file_ending}', pattern_type='sequence', framerate=framerate)
			.output({output_file})
			.run()
			)
			return True
		except:
			return False


	def model_check(self, image):
		if self.model == None:
			return True
		else:
			return self.model.predict(image)[0][0] > 0.5


	def video_to_images(self, mp4_filename, folder_to_write, frame_skip_rate, path = "./"):
		"""
		mp4_filname: video to read from
		path: optional (./ default) to write file in different path
		folder_to_write: name of folder to write imgs to
		frame_skip_rate: how many frames to skip in video (29 = 1fps, 14 = 2fps)
		"""
		vidcap = cv2.VideoCapture(mp4_filename)
		success,image = vidcap.read()
		pred = np.expand_dims(cv2.resize(image, (0,0), fx=0.5, fy=0.5), 0)
		count = 0
		while success:
			if self.model_check(pred):
				cv2.imwrite(f"{path}{folder_to_write}/frame{count}.jpg", image)     # save frame as JPEG file
				count += 1

			success,image = vidcap.read()
			pred = np.expand_dims(
						cv2.cvtColor(
							cv2.resize(image, (0,0), fx=0.5, fy=0.5)
						,cv2.COLOR_BGR2RGB)
					, 0)
			
			for ii in range(frame_skip_rate):
				success,image = vidcap.read()
			







