from setuptools import setup, find_packages
from Cython.Build import cythonize


def read_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        res = f.read()
    return res


install_requires = read_file('requirements.txt').splitlines(False)
classifiers = [
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
]
setup(
    name='fics',
    version='1.0.0',
    license='MIT',
    url='https://github.com/Jintao-Huang/financial-ics/',
    author='Jintao Huang',
    author_email='huangjintao@mail.ustc.edu.cn',
    packages=['fics'],
    install_requires=install_requires,
    classifiers=classifiers,
    python_requires='>=3.8',
    ext_modules=cythonize(
        ["fics/utils/metric/*.pyx"],
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "cdivision": True,
        }),
    zip_safe=False,
)