# pstage_01_image_classification

## Getting Started  

### 주의해서 볼 것
1. TODO 검색해서 Default값 수정해야 할 부분 있는지 확인해보기 (특히 Model이나 Epochs 등은 무조건 수정!)
2. 빠뜨린 TODO가 있을 수도 있기 때문에 자신의 코드랑 Line by Line까지는 아니더라도 개인적으로 다르게 수정한 부분은 주의해서 보기
3. wandb 찍는법

  (1) Console -> wandb login
  
  (2) Wandb에 로그인 하고 Project를 만들면, Proejct 명을 만들어야 할 것. 그 때 Project명을 project에, Entity는 자신의 이름으로 설정하면 됨
  
  ![image](https://user-images.githubusercontent.com/72785706/156280567-9767db1a-30fc-47a9-826c-630f4e859477.png)
  
    * 나는 violetto가 entity, PProject가 Project명으로 지정했기 때문에 init에 그렇게 입력한 것

4. wandb 이름 바꾸기

   (1) wandb 사이트에 직접 들어가 이름 바꾸기
   
   (2) Console 창에서 시작할 때 python train.py 뒤에 `--name {지정할 이름}` 으로 argument 주면 자동으로 { } 사이에 있는 이름으로 Project명 지정됨(default ; exp)

### Dependencies
- torch==1.7.1
- torchvision==0.8.2                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN={YOUR_TRAIN_IMG_DIR} SM_MODEL_DIR={YOUR_MODEL_SAVING_DIR} python train.py`

### Inference
- `SM_CHANNEL_EVAL={YOUR_EVAL_DIR} SM_CHANNEL_MODEL={YOUR_TRAINED_MODEL_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR={YOUR_GT_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python evaluation.py`
