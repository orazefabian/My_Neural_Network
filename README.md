# My_Neural_Network
A neural network which can read a hand written digit

Full neural network which can be trained by setting a learning rate and the amount of training runs.
It then can read a 28 x 28 pixel image and determine which number is written.

gui.py can be used to test the nn dynamically via a simple user interface to draw digits and predict their values

If you want to test the network with test data (png files within data folder), then do this via main.py and make sure to use the already trained weights 
(w0_trained.p, w1_trained.p, w2_trained.p) and set "save weights" to False.
If you want to train your own weights use the prepared w0, w1, w2 files and set "use new weights" mode to True.
You can also save them temporally to these files. 
