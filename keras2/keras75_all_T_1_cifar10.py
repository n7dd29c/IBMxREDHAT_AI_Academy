from keras.applications import VGG16,\
    ResNet50,\
    ResNet101,\
    DenseNet121,\
    InceptionV3, InceptionResNetV2,\
    MobileNetV2,\
    NASNetMobile,\
    EfficientNetB0,\
    Xception
    
################# GAP 쓰기, 기존 최고성능 비교 #################
model_list = [
    VGG16(include_top=False, input_shape=(32,32,3)),
    ResNet50(include_top=False, input_shape=(32,32,3)),
    ResNet101(include_top=False, input_shape=(32,32,3)),
    DenseNet121(include_top=False, input_shape=(32,32,3)),
    # InceptionV3(include_top=False, input_shape=(32,32,3)),
    # InceptionResNetV2(include_top=False, input_shape=(32,32,3)),
    MobileNetV2(include_top=False, input_shape=(32,32,3)),
    NASNetMobile(include_top=False, input_shape=(32,32,3)),
    EfficientNetB0(include_top=False, input_shape=(32,32,3)),
    # Xception(include_top=False, input_shape=(32,32,3)),
]