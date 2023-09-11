### Adding BEiTv2
```
git submodule init
mv beit2_changes.patch unilm
cd unilm
git apply --reject --whitespace=fix beit2_changes.patch
cd ..
ln -s $PWD/unilm/beit2/semantic_segmentation/ $PWD/beit2
```

### Creating environment
```
conda env create -f environment.yml
```

### Install xformers
```
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```