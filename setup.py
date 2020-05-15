import pathlib
from setuptools import setup, find_packages

with open('requirements.txt') as fp:
    install_reqs = [r.rstrip() for r in fp.readlines()
                    if not r.startswith('#') and not r.startswith('git+')]

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(name='soln-ml',
      version='1.0.2',
      long_description=README,
      long_description_content_type="text/markdown",
      description='Soln-ML: Towards Self-Improving AutoML System.',
      author='AutoML Researcher @ DAIM',
      author_email='liyang.cs@pku.edu.cn',
      url='https://github.com/thomas-young-2013/soln-ml',
      keywords='AutoML,machine learning',
      packages=find_packages(exclude=['docs', 'examples', 'test']),
      license='LICENSE.txt',
      test_suite='nose.collector',
      python_requires='>=3.5.*',
      include_package_data=True,
      install_requires=install_reqs)
