## Overview 
This repository contains descriptions and implementations of several "challenge" problems related to the use of inverse problems and data assimilation techniques to characterize sea ice.   By providing these problems, our goal is to accelerate the advancement of algorithms for solving inverse problems that arise in sea ice modeling and remote sensing.  These problems do not capture all of the interesting aspect of this problem space, but should provide an initial bridge between the sea ice community and the computational inverse problem community.

This 
The documentation for these problems can be found at [simda-muri.github.io/challenges/index.html](https://simda-muri.github.io/challenges/index.html).

## Installation

```
python -m pip install git+https://github.com/simda-muri/challenges.git
```

## Notes for developers:

#### Editable Installation
You can install the challenges package with pip in "editing" mode to allow changes to the source code to be used in other scripts.   After cloning the repository, run the following 
```
git clone git@github.com:simda-muri/challenges.git
cd challenges
pip install -e .
```

#### Building Documentation Locally
You can manually run `sphinx` to build the documentation.  From a terminal, `cd` into the docs directory and run `make`:
```
cd challenges/docs
make html
```
If successful, this will produce a bunch of html files in `docs/_build`.  You can view the website by opening `docs/_build/index.html`.

#### Making and Submitting Changes
1. Create a new branch for your work starting from the `main` branch.  Give the branch a short name that describes what you plan to add.
```
git checkout -b my-new-formulation
```
2. Make your changes.   You can update existing files or create new files an folders.  Feel free to commit and push your changes as often as you'd like.  This won't effect the `main` branch or website at all.
3. Submit your changes.   Go to this repository on github and [create a new pull request](https://github.com/simda-muri/challenges/pulls).  Make sure the "base" branch is `main` and the compare branch is the name of your branch, e.g., `my-new-formualation` from above.   Provide a brief description of what you've changed and select someone to review the changes (e.g., Matt or Yoonsang).   Then submit!   

