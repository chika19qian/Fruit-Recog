from keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.08,#横向移动
    height_shift_range=0.08,
    shear_range=0.01,#以弧度逆时针方向剪切角度
    zoom_range=0.1,#随机缩放范围
    horizontal_flip=True,#随机水平翻转
)

gen_data=datagen.flow_from_directory(
    directory='test',#读取目录
    save_prefix='data',#文件开头名
    save_to_dir='new',#存入目录
    target_size=(224,224)#目标尺寸
)
for i in range(5):
    gen_data.next()