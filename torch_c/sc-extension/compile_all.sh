rm -rf build
rm -rf dist
rm -rf sc_extension_cuda.egg-info
rm -rf sc_extension.egg-info
python setup.py install
python setup_cuda.py install
