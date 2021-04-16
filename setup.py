import sys
from pathlib import Path
from setuptools import setup, find_packages

requirements = dict()
for extra in ["dev", "main"]:
    # Skip `package @ git+[repo_url]` because not supported by pypi
    requirements[extra] = [r
                           for r in Path("requirements/%s.txt" % extra).read_text().splitlines()
                           if '@' not in r
                           ]

if sys.version_info < (3, 5, 2):
    raise ValueError("Soln-ML requires Python 3.5.2 or newer.")

if sys.version_info < (3, 6, 0):
    # NumPy 1.19.x doesn't support Python 3.5, only 3.6-3.8.
    requirements['main'].remove('numpy>=1.9.0')
    requirements['main'].append('numpy>=1.9.0,<=1.18.4')

# The directory containing this file
HERE = Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(name='soln-ml',
      version='1.0.5',
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
      install_requires=requirements["main"],
      extras_require={"dev": requirements["dev"]})
