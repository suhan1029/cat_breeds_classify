import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
from torch import nn
import os

# Define the CustomModel class
class CustomModel(nn.Module):
    def __init__(self, base_model, num_classes, dropout):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(460800, num_classes)
       
    def forward(self, x):
        x = self.base_model.features(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Function to load the model
def load_model(model_path, num_classes, dropout):
    base_model = models.efficientnet_b5(weights=None)
    model = CustomModel(base_model, num_classes, dropout)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to predict the breed of the cat
def predict_image(image, model, transform, class_names):
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    return class_names[preds[0]]

# Main function to run the Streamlit app
def main():
    st.title("고양이 품종 분류")
    st.write("고양이 이미지 사진을 올리면 해당 고양이의 품종이 뭔지 분석합니다.")

    # Upload image
    uploaded_file = st.file_uploader("이미지 파일을 업로드하시오", type=["jpg", 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='올린 이미지', use_column_width=True)
        st.write("")

        image_size = 456
        dropout = 0.3
        data_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Define class names
        class_names = ['Abyssinian', 'American Bobtail', 'American Curl', 'American Shorthair',
                       'Bengal', 'Birman', 'Bombay', 'British Shorthair',
                       'Egyptian Mau', 'Exotic Shorthair', 'Maine Coon', 'Manx',
                       'Norwegian Forest', 'Persian', 'Ragdoll', 'Russian Blue',
                       'Scottish Fold', 'Siamese', 'Sphynx', 'Turkish Angora']
        
        face = ['귀엽고 아름다움', '사냥꾼 눈의 야생적인 외모', '까칠하고 사나워 보이는 외모', '크고 동글동글하며 붙임성 있음', 
                '야생적인 외모와 날카로운 턱선', '둥근 머리와 파란 눈을 가진 외모', '코가 짧고 약간 뭉툭하며, 전체적으로 둥글함', 
                '둥글고 큰 얼굴에 넓은 눈과 두꺼운 볼살', '귀가 크고 뾰족함, 중간 크기의 머리와 큰 눈', '납작한 코와 큰 눈, 둥글고 넓은 얼굴', 
                '큰 머리와 뚜렷한 이목구비', '둥근 머리와 큰 눈을 가진 외모', '삼각형 얼굴형과 큰 눈, 전체적으로 강한 인상', 
                '얼굴이 넓고 둥글며, 귀가 작고 둥글게 말려있음', '중간 크기의 머리와 큰 파란 눈', '얼굴이 날렵하고 귀가 크고 뾰족함', 
                '코가 짧고, 얼굴 전체가 둥근 인상', '얼굴이 삼각형 모양이고 코가 길고 뾰족함', '얼굴이 삼각형 모양이고 눈이 크고 주름짐', 
                '코가 길고 날렵하며, 귀가 크고 뾰족함']
        
        personality = ['예민함, 호기심 많음', '태평함, 적응력 MAX', '차분함, 영리함', '낙천적, 쾌활함', 
                       '활발함, 호기심 많음', '온순함, 애정 많음', '낙천적, 애교 많음', '태평함, 차분함', 
                       '활기참, 영리함', '온순함, 낙천적', '친근함, 쾌활함', '활발함, 애정 많음', 
                       '독립적, 차분함', '차분함, 태평함', '온순함, 낙천적', '차분함, 영리함', 
                       '낙천적, 애교 많음', '활기참, 영리함', '활발함, 애정 많음', '영리함, 활기참']
        
        info = ['무엇보다 호기심이 왕성하고 똑똑한 고양이로,\n 귀엽고 아름다운게 당신과 닮았지만 성격은 장점만 닮은걸로?',
                '영리하고 활동적이며 다정한 고양이로,\n 강인한 눈이 당신을 닮았소! 성격도 닮았으면 좋겠소!',
                '호기심이 많고, 동료애가 있으며, 매우 사람을 잘 따르는 고양이로,\n 야생미넘치는 외모가 당신을 닮았소! 그리고 성격도?',
                '온화하고 점잖으면서도 애정이 많은 고양이로,\n 둥근 얼굴이 당신과 닮았소! 혹시 성격도?',
                '활기차고 장난을 많이 치며 주변을 탐험하는 것을 좋아하는 고양이로,\n 강인한 외모가 당신과 닮았다!',
                '사람에게 잘 따르고 조용히 곁에 머무르는 것을 좋아하는 고양이로,\n 안정감을 주는 외모가 당신과 비슷하다!',
                '가족과 함께 놀고 시간을 보내는 것을 즐기는 고양이로,\n 시크한 외모가 당신을 표현한다! 성격도 이 고양이와 닮았으면 이성친구 만들기 쌉가능!',
                '혼자서 조용히 지내는 것을 좋아하며 독립적인 고양이로,\n 귀엽게 생긴 당신의 외모가 부럽다!',
                '빠르고 민첩하게 움직이며 사냥 놀이를 즐기는 고양이로,\n 역삼각인 당신의 외모가 부러울수도?',
                '차분하고 느긋하게 집안을 돌아다니며 휴식을 즐기는 고양이로,\n 도도하면서도 귀여운 얼굴이 당신을 닮았소! 성격은 닮지 않아도 상관 없을지도?',
                '사람들과 잘 어울리고 다양한 활동을 즐기는 고양이로,\n 뚜렷한 이목구비가 당신과 닮았다! 잘생긴 당신이 부럽다!',
                '활발하게 뛰어다니며 놀이와 탐험을 좋아하는 고양이로,\n 무뚝뚝하면서도 순수한 외모가 당신을 닮았소! 혹시 성격도 그런가?!',
                '독립적이지만 가족과 함께 시간을 보내는 것도 좋아하는 고양이로,\n 순수하면서도 사자같은 외모인 당신이 부럽소!',
                '조용하고 느긋하게 집안에서 쉬는 것을 좋아하는 고양이로,\n 귀여운 외모를 가진 당신이 너무 부럽다!',
                '사람에게 잘 따르며 안기거나 옆에 누워 있는 것을 좋아하는 고양이로,\n 도도하면서도 귀여운 것이 당신과 딱 알맞소!',
                '조용하고 침착하게 지내며 관찰하는 것을 좋아하는 고양이로,\n 영리하게 생긴 당신을 보니 세상 참 불공평하다...',
                '놀이를 즐기며 가족과 함께 시간을 보내는 것을 좋아하는 고양이로,\n 순둥순둥하게 생긴게 당신과 찰떡일수도?! 성격까지 닮으면 금상첨화!',
                '활발하고 수다스러우며 사람들과 상호작용하는 것을 좋아하는 고양이로,\n 역삼각인 당신의 잘생긴 외모를 표절하고 있다!',
                '활발하게 움직이며 따뜻한 곳에서 놀고 쉬는 것을 좋아하는 고양이로,\n 공부 잘하게 생겼지만...',
                '호기심 많고 활발하게 집안을 돌아다니며 놀이를 즐기는 고양이로,\n 깨끗하면서 도도한 당신의 외모가 부럽다! 성격도 잘 생긴거면 세상 참...']
        
        # Load model
        model_path = "cat_breeds_efficientnet_b5_83_3H.pth"
        model = load_model(model_path, len(class_names), dropout)

        # Predict breed
        prediction = predict_image(image, model, data_transform, class_names)
        st.write(f"이 고양이의 품종은 {prediction} 입니다.")

        index = class_names.index(prediction)
        st.write(f"**얼굴 특징**: {face[index]}")
        st.write(f"**성격**: {personality[index]}")
        st.write(f"**평소 행실**: {info[index]}")

        # Display example image of the predicted breed
        example_image_path = os.path.join("cat_ex", f"{prediction}.jpg")
        if os.path.exists(example_image_path):
            example_image = Image.open(example_image_path)
            st.image(example_image, caption=f"{prediction} 예시 이미지", use_column_width=True)
        else:
            st.write("예시 이미지를 찾을 수 없습니다.")


if __name__ == "__main__":
    main()
