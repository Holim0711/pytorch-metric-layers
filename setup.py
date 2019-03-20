from setuptools import find_packages, setup


requirements = [
    'setuptools',
    'sklearn',
]


setup(
    name='métrics',
    version='0.1.0',
    license='MIT',
    author='Holim Lim',
    author_email='ihl7029@europa.snu.ac.kr',
    url='https://github.com/Holim0711/pytorch-metric-layers',
    description='Metric layers for pytorch',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
)

