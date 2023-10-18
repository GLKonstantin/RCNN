import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import PIL.Image as ImagePIL

# torch.__version__ = "2.0.1"
print('Версия Torch: ', torch.__version__)
# check GPU
print('GPU доступно: ', 'НЕТ' if torch.cuda.is_available() == False else 'ДА')


class Image:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.tensor = None
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_path} not found")
        else:
            self.load()

    def load(self):
        self.image = ImagePIL.open(self.image_path)
        self.tensor = transforms.ToTensor()(self.image)
        return self.image

    def __str__(self):
        return f"Image: {self.image_path}"

    def __repr__(self):
        return f"Image({self.image_path})"


class ModelX:
    models_folder = os.path.join(os.path.dirname(__file__), "MODELS")
    exist_models = [m for m in os.listdir(models_folder) if m.endswith(".pth")]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, model_name):
        self.mod_name = model_name
        self.mod_path = os.path.join(self.models_folder, model_name + ".pt")
        self.mod = None

        if not os.path.exists(self.mod_path):
            self.load_default()
        else:
            self.load()

    def load_default(self):
        # Загружаем модель с сайта PyTorch Hub. Модель загружаем в папку по умолчанию - self.models_folder
        # Модель для поиска объектов на изображении - Faster R-CNN ResNet-50 FPN
        repo = 'ultralytics/yolov5'
        self.mod_name = 'yolov5s'
        self.mod = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu')
        # Сохраняем модель в папку по умолчанию - self.models_folder
        torch.save(self.mod, self.mod_path)
        self.load()

    def load(self):
        print(f"Loading model {self.mod_name} from {self.mod_path}", os.path.exists(self.mod_path))
        # self.mod = torch.hub.load('MODELS', 'yolov5s', source='local', force_reload=True, path='yolov5s.pt')
        self.mod = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu')

    def predict(self, image: Image):

        with torch.no_grad():
            prediction = self.mod(image.image_path)

        return prediction

    def __str__(self):
        return f"Model name: {self.mod_name}"

    def __repr__(self):
        return f"Model({self.mod_name})"


if __name__ == "__main__":
    m = ModelX("yolov5s")
    img = Image("IMAGES/flights.jpg")
    pre = m.predict(img)
    print(pre)
    print(pre.pandas().xyxy[0])
    print(pre.__dict__)



