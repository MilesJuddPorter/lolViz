# lolViz
Open Source pacakge to extract data from professional league matches! <br>
Contains 4 custom built convolutional neural networks:
1. Classifier for league champion icons
2. Process league clock time (in game time)
3. League digit classifier
4. In game detection (Classifier to determine if image is an active game) - most importantly classifies highlights/replay during stream as not active


# Example use:
```
game = Game(clock_coords = (620, 50, 40, 20))
for img in list_of_imgs:
  game.analyze_image(img)
```
This will process a list of images looking like: <br>
![example_frame](https://user-images.githubusercontent.com/13202373/172957577-44cd1e39-a531-46ee-9b5c-0507ac2945e9.jpg)

<br>
You can then see the champions it predicted: <br>
<img width="614" alt="Screen Shot 2022-06-09 at 6 46 33 PM" src="https://user-images.githubusercontent.com/13202373/172957686-b04d4c51-088f-4bcd-996b-56fe3651da96.png">

<br>
As well as all the data it gathered for every image! <br>
<img width="654" alt="Screen Shot 2022-06-09 at 6 47 06 PM" src="https://user-images.githubusercontent.com/13202373/172957749-eb3e627b-11fe-497a-9af5-1c49b1f2f664.png">

# CLASSES/FILES:
# extractor
This is the main user interaction package <br>
It contains a class: Game() <br>
This class helps to analyze and extract data from league images. It interacts with img_helper and model_helper to save you time!

# img_helper
Contains three classes:
1. LoLImage(full size league image)
2. LoLHud(league hud image) --> can pass in LoLImage.hud
3. LoLMinimap(league minimap image) --> can pass in LoLImage.minimap

# video_helper
This contains a class with 2 main functions: <br>
One to help you turn mp4s into jpg images <br>
Another to help you turn jpgs into an mp4

# model_helper
Classes to serve as a wrapper around a couple custom built models to make life easier

# How it was built
I webscraped over 500 youtube videos of professional matches - about 1 terabyte worth <br>
From there I started building out classes to handle different parts of the screen as well as ways to turn the mp4 into images <br>
I have over 15 sub-folders of things from champion_icon_data_generator to minimap_autoencoder where I built out smaller tools that I could use <br>
From there I trained and trained and trained until I had workable Neural Networks <br>
I then went back and made these NNs usable by the package to extract data!

# Additional Info:
There are tons of things in here I didn't discuss

For example you can view the team hp/mana in LoLImage class <br>
<img width="1792" alt="labeled_im" src="https://user-images.githubusercontent.com/13202373/172958874-f5e44f4a-9c36-4041-a91d-3d5aba27da5e.png">

<a href="https://user-images.githubusercontent.com/13202373/172958036-76a0a9e0-b7be-4faa-8f1f-5105f5f444fb.mp4">Full labeled vid</a>

<br>
And you can also label the minimap with best guesses for champion location:
<img width="654" alt="Screen Shot 2022-06-09 at 6 47 06 PM" src="https://user-images.githubusercontent.com/13202373/172958093-21683f75-9d43-4eeb-bd26-2b5b35def2fd.jpg">
