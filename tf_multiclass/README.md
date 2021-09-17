# TF Multiclass

Multiclass hinge losses for TensorFlow.

Tested with TensorFlow 2.4.1 and 2.5.0.

This package implements the following losses in the permutation-equivariant relative-margin (PERM) loss framework (more on this at the very end):

1. Weston-Watkins hinge. [Reference](https://www.jmlr.org/papers/v17/11-229.html).
2. Crammer-Singer hinge. [Reference](https://www.jmlr.org/papers/v17/11-229.html).
3. Duchi-Ruan-Khosravi hinge. Referred to as the [family-wise loss](https://projecteuclid.org/journals/annals-of-statistics/volume-46/issue-6B/Multiclass-classification-information-divergence-and-surrogate-risk/10.1214/17-AOS1657.full)
and the [adversarial zero-one loss](https://proceedings.neurips.cc/paper/2016/hash/ad13a2a07ca4b7642959dc0c4c740ab6-Abstract.html).
4. Cross entropy.


## How to use

There are examples in the `jupyter_notebooks` directory.

The notebook `02_linear_WWSVM_in_TensorFlow.ipynb` shows how to use TensorFlow to train linear multiclass SVMs.

The notebook `03_comparing_cross_entropy_implementations.ipynb` compares our implementation of the cross entropy with the standard TensorFlow's implementation. (They are probably the same).

## FAQ 

**F**ictitiously **a**sked **q**uestions, since this repository may still be unknown.

### Is this cross entropy the same as THE cross entropy?

By *THE cross entropy*, I am referring to `tf.keras.losses.CategoricalCrossentropy(from_logits=True)`. This is based on *absolute margins*. I will call this the absolute margin CE in contrast to our PERM CE.

So the answer is *probably* yes.
For shallow models, they are equivalent.
See `jupyter_notebooks/03_comparing_cross_entropy_implementations.ipynb` for a demonstration.
It can be proved that relative and absolute margin cross entropies result in the same optimization trajectories for convex models.

However, the PERM CE is strictly convex while the absolute margin CE is not. Whether this leads to any differences when training deep models is, the jury is still out.


### Why another implementation of the cross entropy?

Since the multiclass hinges (DKR, WW, and CS) losses are implemented in the PERM loss framework, it's handy to implement the cross entropy also in this framework. It makes for easy apple-to-apple comparison.
However, in the future, it may be useful to implement the absolute margin versions of these losses as well.


### What is the theory behind PERM loss?

See `jupyter_notebooks/01_introduction_to_PERM_losses.ipynb`.