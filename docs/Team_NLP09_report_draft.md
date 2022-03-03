# Boostcamp AI Tech NLP 9조 (NLPrun) 팀 회고
> 팀원: 김소연, 김준석, 임동진, 최성원, 하성진
> 대회 기간: 2022.02.23~2022.03.04

## 1. 프로젝트 개요
### 프로젝트 목적
- [ ] TODO
### 프로젝트 개요
- [ ] TODO
### 활용 장비 및 재료
* 팀원 전원 개인 서버 사용
* 서버 스펙
	* CPU: Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz, 8 cores
	* 메모리: 88G & 디스크: 100G(+)
	* GPU: Tesla V100-PCIE-32GB
* 협업 툴
	* 커뮤니케이션: slack
	* 코드: github
	* 실험 결과: wandb, tensorboard
### 프로젝트 구조 및 사용 데이터셋의 구조도
* 프로젝트 구조 개요도
* 데이터셋 구조(폴더 구조)
### 기대효과
- [ ] TODO
* 전세계적인 COVID-19 때문에 마스크 착용이 기본 예절화 되어가고 있음. 체온 측정을 위해 얼굴 인식 시스템이 보급화되어가고 
*  나이..성별..에 따른 분석이 왜필요하냐....ㅎ....
## 2 . 프로젝트 팀 구성 및 역할
- [ ] TODO
* 김소연(팀장)
* 김준석
* 임동진
* 최성원
* 하성진

## 3. 프로젝트 수행 절차 및 방법
* 매일 오전, 오후에 줌 미팅을 통해 각자 진행 상황 공유
* 각자 집중하고 싶은 파트 중심으로 작업. 성능 향상에 도움되는 셋팅 공유
* 대회 전날 가장 높은 성능 가진 코드를 베이스라인으로 마지막 날 최종 성능 향상 시도

## 4. 프로젝트 수행 결과
### 탐색적 데이터 분석을 통한 학습 데이터 파악
> **요약**
> 성별(남/여), 나이(18~60세), 마스크 착용여부(미착용, 착용, 잘못된 착용)의 각각 상황에서 클래스 불균형 존재. 따라서, 각각의 조합으로 이루어진 클래스가 전체적으로 불균형 이룸. 
> 60세 이상 데이터가 현저히 작은 것, 나이 분간의 모호성이 존재함으로 나이 예측이 가장 까다로울 것. 
> 전체 데이터셋은 18,900개로 상대적으로 적은 데이터셋. noisy label 일부 존재

* 학습 데이터 Overview
* 각 Feature에 따른 분포와 전체 클래스 분포
* Noisy label : 잘못된 naming
* Training/Validation 나누는 방식에 따른 데이터 분포 + Data leakage 이슈

### 모델 개요
* EfficientNet[1]
	* 기존 네트워크와의 차이점
	* 각 모델 별 인풋 사이즈 및 모델 파라미터(ResNet과 비교한 테이블)

### 모델 선정 이유 및 학습 관련 하이퍼파라미터
> **요약**
> 모델 : Overfitting 문제를 고려해 인풋 사이즈와 파라미터 크기 대비 성능 효율이 좋다고 판단한 EfficientNet 사용(efficientnet b3). classifier에 추가 FC layer와 dropout(0.7) 사용. 
> Metric: F1 score('macro'), accuracy
> 검증 전략: Out-of-fold 앙상블 사용 시 5-fold의 stratified validation 사용. 싱글 모델 경우 4:1로 train:val 데이터셋 분할 후 검증
1. EfficientNet을 사용 이유 : 적은 파라미터를 가질 수록 모델의 복잡도가 낮아지고 Overfitting 을 피할 수 있기 때문
	* 해당 데이터는 적은 데이터셋과 특정 클래스에 overfitting될 수 있는 이슈 존재.
	* ResNet, VGG, Densenet의 경우 parameter 갯수가 많음. 최근 SOTA를 찍고 있는 Vision Transformer 계열은 Downstream task 적용 시 상대적으로 많은 데이터셋이 필요하기 때문에 고려하지 않음.
	* Overfitting을 처리할 수 있는 방법 중 모델의 복잡도를 줄이는 방향으로 설계.
	* EfficientNet의 성능대비 파라미터가 적기 때문에 가장 적은 efficientnetb0(5.6M)~efficientnetb4( xxM) 로 실험 수행. 
	
2. Modified EfficientNet : 
	* 기존 모델에서 final classifier 만 updating 시킬 경우 natural dataset인 imagenet에 기훈련된 모델의 representation power 향상이 부족할 수 있을 것이라 판단. 추가 classifier를 사용, dropout으로 regularization 적용

3. Single model 기준 전체 학습에 사용한 셋팅 
	 * LB 점수 : 0.74(F1), 0.78(ACC)
	 * input image size: 
	 * Batch size:
	 * Augmentation:
		 * Training: 
		 * Validation
	 * Loss: Cross Entropy loss
	 -  [  ] TTA
	 * Optimizer/lr/weight dcay/ scheduler: Adam, 1e-4, 5e-5, decaying 0.995 at every epoch
	 * 추가 시도
		 * multi-dropout? ensemble 대신?
		 * optimizer 변경
		 *  Loss 변경(todo LDMA)

4. Ensemble 기준 전체 학습에 사용한 셋팅 (TODO)
	 * LB 점수 : 0.74(F1), 0.78(ACC)
	 * input image size: 
	 * Batch size:
	 * Augmentation:
		 * Training: 
		 * Validation
	 -  [  ] TTA
	 * Optimizer/lr/weight dcay/ scheduler: Adam, 1e-4, 5e-5, decaying 0.995 at every epoch
	 * 추가 시도
		 * multi-dropout? ensemble 대신?
		 * optimizer 변경
 4. 검증 전략
	 * Metric: F1 score
	 * 검증 전략 : 
	 * 앙상블 방법 : soft voting/ hard voting
	 
### 모델 성능 평가 및 개선 
1. 싱글 모델 평가 
	* 모델 Training acc, loss와 Validation acc, loss : 모델에 상관 없이overfitting을 겪는 현상.. 어느정도 overfitting이 있을것이라 보여지는 epoch 쯤(6~8) 에서 제출 +피겨 추가 -> 어떻게 early stopping한 것인지 추가되면 좋겠다!
	* Confusion Matrix(Epoch 별 사진 첨부) : epoch 별 age 구간에서 prediction이 헷갈리는 것을 알 수 있음

