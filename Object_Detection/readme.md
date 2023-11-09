# Candy Object Detection  
## The image data (10 candy pictures) are manually labelled with label-studio, and the model is trained on the 10 images following HuggingFace object detection standard
### The labelling result (COCO formatted annotations Json) is in the result.json, and it needs to further cleaned in the code to generate the desired format that is readable by dataset.load_dataset(), which is a HuggingFace API

More information:
HuggingFace Object Detection Tutorial: https://huggingface.co/docs/transformers/tasks/object_detection

Label Studio Quickstart: https://labelstud.io/guide/get_started#Quick-start