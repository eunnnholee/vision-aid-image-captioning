# Image Captioning : 시각장애인을 위한 보조 기능

## I. 프로젝트 개요

### 연구목적:
> - 시각장애인에게 도움을 줄 수 있는 방법: Image Captioning
> - 강의에서 배운 내용을 합쳐 응용해볼 수 있는 Multi Modal Task
> - 여러 모델을 비교 분석하면서 성능을 정성적으로 test


### 설치 패키지
```
pip install tensorflow
pip install pandas
pip install numpy
pip install matplotlib
pip install requests
pip install Pillow
pip install tqdm
pip install nltk
pip install torch torchvision
```

</br>

## II. Image Captioning Model
> - CNN을 이용해 이미지 특징(feature)을 추출한 뒤 이미지 특징을 입력으로 RNN에 주어 문장을 생성하는 모델
> - 기계 번역(machine translation) 작업(Seq2Seq)에서의 인코더를 CNN으로 대체한 것과 같다.

</br>

### CNN Image Embedding: Inception V3
> - 다양한 크기의 커널을 동시에 사용한 Inception 모듈을 사용하여 더욱 유연하게 특징 추출
>   - 1x1 convolution을 활용한 channel 수 감소 -> 연산량↓
>   - 합성곱 분해-> 연산량↓

</br>

### Decoder : LSTM and Transfomer의 비교
> 기존의 이미지의 특징을 추출하는 CNN과 단어를 생성하는 LSTM 기반의 이미지 캡션 모델과 쫌 더 고도화되고 최신기법인 Inception V3와 Transfomer를 활용해 단어를 생성하는 이미지 캡션 모델을 정성적으로 비교하고자 했다.

--- 

</br>

### 데이터셋:
> - [Kaggle의 flickr8 Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
>   - 해당 데이터셋은 사진과 사진에 대한 설명(Caption)이 쌍으로 이루어진 데이터셋
>   - 규모: 40,400장 (1.12GB)
>   - Image Captioning의 데이터셋은 대부분 COCO 데이터셋과 COCO caption인 약 27GB의 정도의 규모의 데이터로 학습하는 것이 일반적이지만, 해당 연산을 시행할 컴퓨팅 리소스가 부족하여, 작은 데이터셋 골라 진행했다.

</br>

### Train / Test Split
> - Train: Image - caption pair = 32,634
> - Test: Image - caption pair = 8,091

</br>

### 전처리 : 데이터 증강
> - 사진이 노이즈가 끼어있찌 않고 캡셔닝 또한 데이터가 정형화되어 있기 때문에 이미지에 대해서, Resize를 제외한 특별한 전처리를 진행할 필요가 없었다.
> - 일반화 성능을 높이기 위해 이미지 증강을 진행했다.
>   - 이미지 수평으로 좌우 반전
>     - 수직으로 반전하는 경우 도메인적으로 이미지의 맥락이 훼손될 수 있다고 판단하였기에 진행하지 않았다.
>   - 이미지 랜덤으로 기울이는 Random Rotate 진행
>   - 픽셀값의 차이를 크게 만드는 contrast 진행
```
image_augmentation = tf.keras.Sequential(
[
tf.keras.layers.RandomFlip("horizontal"),
tf.keras.layers.RandomRotation(0.2),
tf.keras.layers.RandomContrast(0.3),
]
```


</br>

### 토큰화 및 임베딩:
> - 단어가 최소 4번 이상 등장해야, Tokenizer에 토큰화, threshold=4
> - 등장 횟수가 4보다 작을 시 Pad(unknown) 토큰 처리
> - [Start], [End] 토큰 부여
> - SOS : Start Of Sentence
> - EOS : End Of Sentence
<img width="531" alt="image" src="https://github.com/eunnnholee/vision-aid-image-captioning/assets/151797888/2d3d6f14-8326-471c-80f2-b19203c77657">

<img width="442" alt="image" src="https://github.com/eunnnholee/vision-aid-image-captioning/assets/151797888/0f2e08e6-0b5f-475d-9189-af38164446de">

</br>
</br>

### Modeling
> 개인의 GPU를 이용하여 학습
> - 과적합 방지: Early Stopping
> - ImageNet 데이터를 통해 Pretrained 된 CNN(ResNet, InceptionV3) 사용하여 Latent Vector를 추출
```
def CNN_Encoder():
inception_v3 = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

output = inception_v3.output output = tf.keras.layers.Reshape((-1, output.shape[-1]))(output)

cnn_model = tf.keras.models.Model(inception_v3.input, output)
return cnn_model
```
> - include_top = False로 설정하여,모델의 (fully connected) 레이어를 포함하지 않도록 설정했다. 원래의 모델에서 마지막 레이어는 이미지 분류를 위한 것이지만, 이미지의 특징을 추출하기 위해 이를 사용하지 않았다.
> - output의 형태를 tf.keras.layers.Reshape 레이어를 사용하여 출력을 3D 텐서에서 2D 텐서로 변환했다. Transformer 모델의 Encoder의 Input으로 사용하기 위함이다.

</br>

하이퍼파라미터 설정
```
1. Transformer의 Embedding Dim : 주어진 이미지와 캡셔닝의 복잡도를 고려하여 2의배수 중 일반적인 하이퍼 파라미터 값인 512로 설정
2. Transformer의 Unit(head) : 보편적인 값인 512로 설정
3. Learning Rate : keras adam 옵티마이저의 기본값, 0.001로 설정
4. Early Stopping : Patience=3으로 설정하여, 3번이상 Loss가 줄어들지 않는다면 학습을 중단하도록 설정
```

</br>

### 학습결과
> 손실 함수는 각 위치의 단어를 올바르게 예측할 수 있도록 하기 위해, 분류 문제에서 정답 클래스를 정확하게 분류할 확률을 높이는 Cross Entropy Loss로 설정
> - Transformer와, LSTM모델 모두 5번의 Epoch을 통해 학습을 모두 진행 했으며 아래는 Transfomer 모델의 Loss를 Plot한 결과이다.
<img width="334" alt="image" src="https://github.com/eunnnholee/vision-aid-image-captioning/assets/151797888/b1f2485d-d3c1-4643-871f-9f4435eaa59e">

</br>

<img width="474" alt="image" src="https://github.com/eunnnholee/vision-aid-image-captioning/assets/151797888/864a2733-ffa5-475d-bc1e-dec79e1c8409">
- LSTM을 사용한 캡셔닝 모델은 "playing in water"를 정확하게 인식했지만, 주어인 "dogs"를 잘못 이해하여 정확한 캡션을 생성하지 못한다.
- 반면, 트랜스포머 모델은 주어로서 "사람이 물 위에 서있다"는 문맥을 잘 파악하여 더 의미 있는 캡션을 생성한다.
> 해당 사진과 문장이 일반적이지 않은 특별한 상황임을 감안할 때, 트랜스포머 모델은 꽤나 탁월한 성능을 보이고 있는 것으로 판단된다.

</br>

<img width="466" alt="image" src="https://github.com/eunnnholee/vision-aid-image-captioning/assets/151797888/de402b3d-4602-44d7-926d-5016f015ee60">
- 트랜스포머 모델이 모든 문장에서 탁월한 성능을 보인 것은 아니다.
- 특히, "hockey", "frozen pond"와 같이 특이한 상황에서는 모델이 안전한 선택을 위해 일반적이고 흔한 단어들을 선호하는 경향이 있다.
- 효과적인 모델 성능을 위해서는 더 다양하고 특이한 상황에 대한 학습 데이터가 더 필요할 것으로 판단된다.

</br>
</br>

## III. 결론
> 동일한 데이터셋으로 학습하였음에도 불구하고 캡셔닝 모델의 디코더에 따라 LSTM과 Transformer 간의 성능 차이가 크게 나타났다. 그러나 4만장의 문장과 사진 쌍으로는 특수한 상황에 대한 학습이 충분하지 않아 실험의 한계로 간주된다.

</br>

### 고도화 방향
> 이미지 캡셔닝 결과를 텍스트로 출력하고, 그 텍스트를 음성으로 변환하여 재생
```
from gtts import gTTS
from IPython import display

test_image = '/Users/eunholee/Downloads/deeplearning_term/prof_yoon.jpg'

captions = beam_evaluate(test_image)

print(f"your model's caption : {captions}")

# Converting text to speech
tts = gTTS(captions, lang='en', slow=False)

# This can be downloaded and played
filename = '/Users/david/Downloads/deeplearning_term/caption.mp3'
tts.save(filename)
display.display(display.Audio(filename, rate=None, autoplay=False))  # To display playback bar as output

>>> your model's caption : an man in a blue
```
Ground Truth : Profile of the Greatest Professor in Yonsei
</br>
- 개인 신상으로 인해 교수님 사진은 따로 첨부하지 않았습니다.
- 모델의 캡션 결과인 'an man in a blue'를 gTTS library를 통해 텍스트를 음성으로 변환


