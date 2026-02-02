import os
import subprocess
import time
from setuptools import find_packages, setup

def get_git_commit_number():
    if not os.path.exists('.git'):
        num = str(time.time())[0:6]
        return num

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number

def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':
    version = '1.0.0+%s' % get_git_commit_number()
    write_version_to_file(version, 'uavdet3d/version.py')


    setup(
        name='uavdet3d',
        version=version,
        description='Open3DUAVDet is a general codebase for 3D UAV detection from multi-modal data',
        install_requires=[
            'numpy',
            'tensorboardX',
            'easydict',
            'pyyaml',
            'tqdm',
            'torch',
            'torchvision',
            'pandas',
            'matplotlib',
            'scipy',
            'opendv-python',
            'transforms3d',
        ],
        author='Hai Wu',
        author_email='wuhai@stu.xmu.edu.cn',
        license='Apache License 2.0',
        packages=find_packages(exclude=['tools', 'data', 'output']),
    )

