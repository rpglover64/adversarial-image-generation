# Notes

* I didn't bother setting up a package. If necessary, my first impulse would be to use Poetry
* I picked inception v3 to target
* I decided to figure out how to do this myself rather than look up a paper explaining how to do it well; I've probably read about it before, so I'm not claiming to have figured it out from scratch, but not recently, so I'm not explicitly following a known-good methodology.

## V1

* Used SGD optimizer with default parameters and some L2 regularization for fixed number of iterations
* Initialized to noise to zeros and targeted a fixed class
* resulting image has distinctive visual artifacts

Noise after optimization:

![noise after optimization](<V1 artifacts/noise.png>)

Image with noise:

![image after noise addition](<V1 artifacts/perturbed.png>)

Next attempt:

* parameters rather than hard code most things
* Experiment with stronger regularization, more training iterations, and lower initial learning rate
