#!/usr/bin/env python

#!/usr/bin/env python
import os
from setuptools import setup
import warnings
import argparse

# https://stackoverflow.com/questions/18725137/how-to-obtain-arguments-passed-to-setup-py-from-pip-with-install-option


try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--local", action="store_true", help="use local file")
args, unknown = parser.parse_known_args()
print(args)

def get_requirements(path="requirements.txt"):
    """
    Return requirements as list.

    package1==1.0.3
    package2==0.0.5
    """
    with open(path) as f:
        packages = []
        for line in f:
            package = line.strip()

            # let's also ignore empty lines and comments
            if not package or package.startswith('#'):
                continue
            if 'https://' in package:
                tail = package.rsplit('/', 1)[1]
                tail = package.split('#')[0]
                package = package.replace('@', '==').replace('.git', '')
            if package.startswith('-r'):
                # recursive requirements
                packages.extend(get_requirements(package.split(' ',1)[1]))
            elif package.startswith('-e'):
                # editable requirements
                _, package = package.split(' ',1)[1]
            elif package.startswith('--'):
                continue
            if "@" in package and args.local: # look for local copy
                pre, post = package.split("@",1)
                local_path = "../" + pre.strip()
                if os.path.exists(local_path):
                    warnings.warn(f"Using local version of {pre} ({local_path})")
                    package = local_path
            if package.startswith('.') or package.startswith('/'):
                package_name = os.path.basename(package)
                package = f"{package_name} @ file://localhost" + os.path.abspath(package)

            if "file://." in package: # this won't work with pip -r
                pre, url = package.split("file://")
                url = os.path.abspath(url)
                package = pre + "file://localhost" + url
            packages.append(package)
    print(f"PACKAGES: {packages}")
    return packages

setup(name='char_tokenizer',
      version='0.0.1',
      description='char_tokenizer',
      long_description= "" if not os.path.isfile("README.md") else read_md('README.md'),
      author='Taylor Archibald',
      author_email='taylor.archibald@byu.edu',
      url='https://github.com/tahlor/char_tokenizer',
      setup_requires=['pytest-runner',],
      tests_require=['pytest','python-coveralls'],
      packages=['char_tokenizer'],
      install_requires=[
          get_requirements(),
      ],
     )
