# lolViz
Open Source pacakge to extract data from professional league matches! <br>
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


# Example use:
```
game = Game(clock_coords = (620, 50, 40, 20))
for img in list_of_imgs:
  game.analyze_image(img)
```
This will process a list of images looking like:
![example_frame](https://user-images.githubusercontent.com/13202373/172957577-44cd1e39-a531-46ee-9b5c-0507ac2945e9.jpg)

You can then see the champions it predicted:
<img width="614" alt="Screen Shot 2022-06-09 at 6 46 33 PM" src="https://user-images.githubusercontent.com/13202373/172957686-b04d4c51-088f-4bcd-996b-56fe3651da96.png">

As well as all the data it gathered for every image!
<img width="654" alt="Screen Shot 2022-06-09 at 6 47 06 PM" src="https://user-images.githubusercontent.com/13202373/172957749-eb3e627b-11fe-497a-9af5-1c49b1f2f664.png">


