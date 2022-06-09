# lolViz
Open Source pacakge to extract data from professional league matches!
Contains 3 custom built convolutional neural networks:
1. Classifier for league champion icons
2. Process league clock time (in game time)
3. League digit classifier

# extractor
This is the main user interaction package
It contains a class: Game()
This class helps to analyze and extract data from league images. It interacts with img_helper and model_helper to save you time!

# img_helper
Contains three classes:
LoLImage(full size league image)
LoLHud(league hud image) --> can pass in LoLImage.hud
LoLMinimap(league minimap image) --> can pass in LoLImage.minimap

# video_helper
This contains a class with 2 main functions:
One to help you turn mp4s into jpg images
Another to help you turn jpgs into an mp4

# model_helper
Classes to serve as a wrapper around a couple custom built models to make life easier
