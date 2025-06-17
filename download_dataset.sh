wget "https://drive.google.com/uc?export=download&id=11DI22zKWtHjXMnNGPWNUbyGz-JiEtZy6" -O "dataset.zip"
unzip "dataset.zip" -d "Data"
rm "dataset.zip"
# ################################################################################
wget "https://drive.google.com/uc?export=download&id=1IqwE0wbCT1orVdZpul2xFiNkGnYs4t89" -O "Data/datasets/EEG_Eye_State.arff"

# CUDA_VISIBLE_DEVICES=7 python DiffusionTS_base.py --preffix daghar_kh --dataset_name KuHar --scaler minmax --class_id 0