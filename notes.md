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

## V2

* Large learning rate with strong regularization seems to give an invisible result
* The target class becomes most likely, but not 99% likely

Noise after optimization:

![noise after optimization](<V2 artifacts/noise.png>)

Image with noise:

![image after noise addition](<V2 artifacts/perturbed.png>)

If I spent more time:

* I kept the debug printing in; I would remove it
* I would consider using a standard training loop rather than an ad-hoc one
* I would look for ways to avoid the noise becoming visible
  * It's mostly in dark areas of the picture; this leads to two immediate directions of exploration:
    * Avoid adding any noise to the dark sections of the image
    * Use multiplicative rather than additive noise
  * Add a regularization term that penalizes based on perceptual distance rather than just noise magnitude
  * Use a different parameterization of noise, such as adding it to the Fourier domain
* The ideal number of iterations depends on the image and target class; it shouldn't be too hard to find a better stopping criterion
* SGD with a fixed step size is definitely not the best optimizer for this; at least, I'd explore using line search
* Testing: For multiple starting images and multiple target classes, run the adversarial training; each one should successfully fool the classifier and remain invisible
