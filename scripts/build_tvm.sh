num_cpus=4

export $TVM_HOME
echo $TVM_HOME
cd $TVM_HOME
git pull
cd build
cmake ../
make -j $num_cpus
cd ../python
sudo python setup.py install --user