from setuptools import setup

setup(name='metattack',
      version='0.1',
      description='Adversarial Attacks on Graph Neural Networks via Meta Learning',
      author='Daniel Zügner, Stephan Günnemann',
      author_email='zuegnerd@in.tum.de',
      packages=['metattack'],
      install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'tensorflow'],
zip_safe=False)