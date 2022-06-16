from setuptools import setup

setup(
    name='SIMDA Challenges',
    version='0.1.0',    
    description='Descriptions and code for sea ice modeling and data assimilation (SIMDA) challenge problems.',
    url='https://github.com/simda-muri/challenges',
    author='Matthew Parno',
    author_email='matthew.d.parno@dartmouth.edu',
    license='BSD 3-clause',
    packages=['simda'],
    install_requires=['matplotlib',
                      'numpy',   
                      'pandas',
                      'pydata-sphinx-theme',
                      'sphinx-panels'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.10',
    ],
)