## CVTest - Experimental Version of Automap for Pixelblaze
Copyright 2024 ZRanger1.

**NOTE: This repository contains the last experimental OpenCV/Python version of Automap before I decided to go all
in on machine learning and the phone app thing.  It will not be regularly maintained and is provided as-is.**

Automap uses a webcam to map LED positions in 2D.  It works with the Pixelblaze to step sequentially through each LED,
and generates a map file which can be imported into the Pixelblaze mapping tab.

### This code implements some of the core Automap ideas:
- it is fairly robust with regard to lighting conditions, even if they change during a run
- it implements adaptive thresholding and contour detection for improved detection reliability
- it works reasonably well with a variety of diffusers and LED types (I've tested LEDs diffused with paper, cloth, glass beads, fake rabbit fur, etc.)
- if, for some reason, it can't find a pixel, it will mark it with coordinates [-1,-1] in the map file so you can find it easily.
- you can rotate the generated map file to match the physical orientation of your LEDs. (No quantization or snap-to-grid though.)

### The Downside:
- You'll want to be comfortable with Python to get the most out of this.  Some editing of parameters may improve your results in certain situations.
- It's slower than I'd like. This is due to very slow frame capture in OpenCV's VideoCapture implementation.
- UI. There's kind of a UI, but it's just enough to support my experiments.  Nothing you'd consider polished.
- The camera must remain still during the mapping process.  (The phone version won't require this.)
- if your computer has more than one camera, you'll need to edit the code to select the correct one.

### Dependencies:
- Python 3.10 or later
- opencv-python
- numpy
- pixelblaze-client

You'll also want to install the Automap.epe pattern (included in the Pixelblaze folder) on your Pixelblaze.

To Run:
- be sure your Pixelblaze is on, with the Automap pattern running
- run the script: ```python CVTest.py <Pixelblaze IP Address>```
- follow the instructions on screen.

 


 

