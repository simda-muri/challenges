## Overview 
This repository contains descriptions and implementations of several "challenge" problems related to the use of inverse problems and data assimilation techniques to characterize sea ice.   By providing these problems, our goal is to accelerate the advancement of algorithms for solving inverse problems that arise in sea ice modeling and remote sensing.  These problems do not capture all of the interesting aspect of this problem space, but should provide an initial bridge between the sea ice community and the computational inverse problem community.


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
