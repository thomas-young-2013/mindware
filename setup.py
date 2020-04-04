from setuptools import setup, find_packages

with open('requirements.txt') as fp:
    install_reqs = [r.rstrip() for r in fp.readlines()
                    if not r.startswith('#') and not r.startswith('git+')]

setup(name='automlToolkit',
      version='1.0',
      description='AutoML toolkit',
      author='AutoML Researcher @ DAIM',
      author_email='liyang.cs@pku.edu.cn',
      url='https://github.com/thomas-young-2013/',
      keywords='AutoML',
      packages=find_packages(exclude=['docs', 'examples', 'test']),
      license='LICENSE.txt',
      test_suite='nose.collector',
      python_requires='>=3.5.*',
      include_package_data=True,
      install_requires=install_reqs)
