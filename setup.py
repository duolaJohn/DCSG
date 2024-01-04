from setuptools import setup, find_packages


setup(name='dcsg',
      version='1.0.0',
      description='',
      author='',
      author_email='',
      url='',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu==1.6.3'],
      packages=find_packages()
      )