---
license: apache-2.0
---
# human-perception-place-pulse

## What does the model do
### safety, lively, beautiful, wealthy, boring and depressing.
Getting human perception scores from street-level imagery. 

The scores are in scale of 0-10.

` Safety, lively, beautiful, wealthy`  high score indicates strong **positive** feeling

` Boring, depressing`  high score indicates strong **negative** feeling

## Model
The models are pre-trained on MIT Place Pulse 2.0 dataset. The backbone of the model is vision transformer pretrianed on ImageNet (ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1). 
3 Linear layers are added in ViT heads for classification.

## How to run the model
Install packages from requirements.txt

` pip install -r requirements.txt` 

Change the file path in *eval.py*

```
model_load_path = "./model/"   # model path
images_path = "./test_image"      # your input image path
out_Path = "./output"     # output path
```
Run the file *eval.py*

`python eval.py`

## References
Please refer to [human-perception-place-pulse](https://github.com/strawmelon11/human-perception-place-pulse) for details.