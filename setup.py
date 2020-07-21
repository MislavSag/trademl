from setuptools import find_packages, setup

setup(
    name='trademl',
    version='0.1.2',
    license='MIT',
    author='Mislav Sagovac',
    author_email='mislav.sagovac@contenio.biz',
    url='https://github.com/MislavSag/trademl',
    packages=['trademl', 'trademl.modeling'],
    include_package_data=True
)
