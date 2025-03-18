from setuptools import setup
from Cython.Build import cythonize
import os
import shutil
import sys


def copy_folder(from_dir_path, to_dir_path):
	if os.path.exists(to_dir_path):
		shutil.rmtree(to_dir_path)
	shutil.copytree(from_dir_path, to_dir_path)

v=sys.version.split()[0]
v=v.split('.')
if len(v)<3:
	print('Error:python version format!')
	sys.exit(1)
num=''.join(v[:2])
#folder_path='build/lib.linux-x86_64-3.7'
filenames=['runYOLO','runDetection','runOCR']
package='packagesShipCode'
requirement=os.path.join(package,'requirements.txt')

if not os.path.exists(package):
	os.mkdir(package)

os.system("pip freeze > %s"%(requirement))
for filename in filenames:
	setup(
	    name="py2c",
	    ext_modules=cythonize('%s.py'%(filename)),
		description="Project to detect ships",
	    author="FJJ"
	)
	cfile = filename + '.c'
	if os.path.exists(cfile):
		os.remove(cfile)
		print("删除%s" % cfile)

buildfolder='build'
subfolder=os.listdir(buildfolder)
folder_path=None#生成路径
for sub in subfolder:
	if sub.startswith('lib'):
		folder_path=os.path.join(buildfolder,sub)
		break

sofiles=os.listdir(folder_path)
for file in sofiles:
	srcPath = os.path.join(folder_path ,file)
	dstPath = os.path.join(package, file)
	print(srcPath)
	shutil.copy(srcPath,dstPath)
	#删除中间多余文件
	if os.path.exists(file):
		os.remove(file)
		print("删除%s" % file)

shutil.copy('run.py',os.path.join(package,'run.py'),)
shutil.copy('ReadMe.txt',os.path.join(package,'ReadMe.txt'))
copy_folder("detModel", os.path.join(package,'detModel'))
copy_folder("TestData", os.path.join(package,'TestData'))
copy_folder("fonts", os.path.join(package,'fonts'))
#os.remove(srcPath)
#os.remove('%s.cpython-%sm-x86_64-linux-gnu.so' % (filename,num))




