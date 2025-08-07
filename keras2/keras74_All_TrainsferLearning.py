from keras.applications import VGG16, VGG19,\
    ResNet50, ResNet50V2,\
    ResNet101, ResNet101V2, ResNet152, ResNet152V2,\
    DenseNet201, DenseNet169, DenseNet121,\
    InceptionV3, InceptionResNetV2,\
    MobileNet, MobileNetV2,\
    MobileNetV3Small, MobileNetV3Large,\
    NASNetMobile, NASNetLarge,\
    EfficientNetB0, EfficientNetB1, EfficientNetB2,\
    Xception
    
model_list=[VGG16(), VGG19(), ResNet50(), ResNet50V2(), ResNet101(), ResNet101V2(),
            ResNet152(), ResNet152V2(), DenseNet201(), DenseNet169(), DenseNet121(),
            InceptionV3(), InceptionResNetV2(), MobileNet(), MobileNetV2(),
            MobileNetV3Small(), MobileNetV3Large(), NASNetMobile(), NASNetLarge(),
            EfficientNetB0(), EfficientNetB1(), EfficientNetB2(), Xception()]

for model in model_list:
    model.trainable = False
    print('\n==============================================')
    print(model.name)
    print(len(model.weights))
    print(len(model.trainable_weights))