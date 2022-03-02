# pstage_01_image_classification
> Last updated by sykim 0302 12:50pm
## Getting Started  
### F1 score
1. sklearn 없으면 'pip install -U scikit-learn' 으로 설치
2. train.py 파일에 training evalutation 부분에 넣고, wandb로 트래킹 되게 했음.
3. best model file을 저장하기 위해(또는 early stopping) train.py에서 best_val_acc로 판단하는데, 그걸 f1_score로 바꾸던지 알아서 취사선택
```python
   # 여기서 기준을 val_acc 말고 f1_result_mac으로 할거면 그거에 따라서 수정
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
```
### Model
1. model.py 에 `ModifiedEfficientB0` 넣어뒀음. import해서 사용
2. timm 라이브러리 설치 및 사용가능한 모델 찾아보기 -> https://rwightman.github.io/pytorch-image-models/

### 사용한 Transformation 및 configuration
* Transformation : efficientnet-b0 기준
```python
    train_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
  
    val_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                        transforms.ToTensor()])
```
* batch size: 64
* optimizer: adam, lr: qe-4, lr scheduling: 매 에폭별 0.995씩 떨어뜨 
```python
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=opt,
                                            lr_lambda=lambda epoch: 0.995 ** epoch,
                                            last_epoch=-1,
                                            verbose=False)
```
* 스케줄러는 위와 같이 선언하고 `scheduler.step()` 해야하고, 본 베이스라인 코드의 line 228 에 구현돼있음. 스케줄러 종류만 바꿔보면서 하면 될듯




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
   
   (2) Console 창에서 시작할 때 `python train.py` 뒤에 `--name {지정할 이름}` 으로 argument 주면 자동으로 { } 사이에 있는 이름으로 Project명 지정됨(default ; exp)

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
