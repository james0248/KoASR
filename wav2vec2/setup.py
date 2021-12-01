# nsml: fifteenhundred/koasr:v2
from distutils.core import setup

setup(name='ladder_networks',
      version='1.0',
      install_requires=[
          'datasets==1.15.1',
          'transformers==4.12.3',
          'torch==1.10.0',
          'librosa',
          'jiwer==2.3.0',
          'hangul_utils',
          'gdown',
          'apex',
          'hunspell'
      ])
