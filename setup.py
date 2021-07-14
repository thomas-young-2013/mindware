import sys
import importlib.util
from pathlib import Path
from setuptools import setup, find_packages

requirements = dict()
for extra in ["dev", "main"]:
    # Skip `package @ git+[repo_url]` because not supported by pypi
    requirements[extra] = [r
                           for r in Path("requirements/%s.txt" % extra).read_text().splitlines()
                           if '@' not in r
                           ]

if sys.version_info < (3, 6, 0):
    raise ValueError("MindWare requires Python 3.6 or newer.")

# if sys.version_info < (3, 6, 0):
#     # NumPy 1.19.x doesn't support Python 3.5, only 3.6-3.8.
#     requirements['main'].remove('numpy>=1.9.0,<1.20.0')
#     requirements['main'].remove('lightgbm')
#     requirements['main'].append('numpy>=1.9.0,<=1.18.4')
#     requirements['main'].append('lightgbm<3.2.0')

# Find version number
spec = importlib.util.spec_from_file_location("mindware.pkginfo", str(Path(__file__).parent / "mindware" / "pkginfo.py"))
pkginfo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pkginfo)
version = pkginfo.version
package_name = pkginfo.package_name


# The directory containing this file
HERE = Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(name=package_name,
      version=version,
      long_description=README,
      long_description_content_type="text/markdown",
      description='MindWare: Towards Efficient AutoML System.',
      author='AutoML Researchers @ DAIR',
      author_email='liyang.cs@pku.edu.cn',
      url='https://github.com/thomas-young-2013/mindware',
      keywords='AutoML; Machine Learning; Deep Learning',
      packages=find_packages(exclude=['docs', 'examples', 'test']),
      license="MIT",
      install_requires=requirements["main"],
      extras_require={"dev": requirements["dev"]},
      package_data={"mindware": ["py.typed"]},
      include_package_data=True,
      python_requires='>=3.6.0',
      test_suite='nose.collector',
      entry_points={
        "console_scripts": [
            "mindware = mindware.__main__:main",
        ]
      }
)
