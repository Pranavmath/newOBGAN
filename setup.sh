apt-get install unzip vim -y
git clone https://github.com/Pranavmath/newOBGAN.git
pip install wandb torchmetrics[detection]
mv refineddataset.zip newOBGAN/refineddataset.zip
unzip newOBGAN/refineddataset.zip
ls