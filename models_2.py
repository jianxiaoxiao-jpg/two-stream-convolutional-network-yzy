# coding: utf-8
# In[1]:
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
import tensorflow as tf
from objectives import cca_loss

def cca_attention_1(predict1, predict2):
    """RGB分支注意力机制"""
    predict1 = tf.nn.sigmoid(predict1)
    split_tensors = tf.split(predict1, num_or_size_splits=2, axis=1)
    tensor1 = split_tensors[0]
    expanded = tf.expand_dims(tf.expand_dims(tensor1, 1), 1)
    return Multiply()([predict2, expanded])

def cca_attention_2(predict1, predict2):
    """光流分支注意力机制""" 
    predict1 = tf.nn.sigmoid(predict1)
    split_tensors = tf.split(predict1, num_or_size_splits=2, axis=1)
    tensor2 = split_tensors[1]
    expanded = tf.expand_dims(tf.expand_dims(tensor2, 1), 1)
    return Multiply()([predict2, expanded])

# 修改1：调整concat函数参数接收方式
def concation(inputs):
    """特征拼接函数（显式接收列表输入）"""
    predict1, predict2 = inputs
    return tf.concat([predict1, predict2], axis=-1)

def create_model(outdim_size, use_all_singular_values):
    # ===================== 双输入分支 =====================
    # RGB输入分支
    img_input1 = Input(shape=(224, 224, 3))
    x0 = Conv2D(64, (3,3), padding='same', activation='relu', name='mod0_block1_conv1')(img_input1)
    x0 = Conv2D(64, (3,3), padding='same', activation='relu', name='mod0_block1_conv2')(x0)
    x0_pool = MaxPooling2D((2,2), strides=(2,2), name='mod0_block1_pool')(x0)

    # 光流输入分支 
    img_input2 = Input(shape=(224, 224, 18))
    x1 = Conv2D(64, (3,3), padding='same', activation='relu', name='mod1_block1_conv1')(img_input2)
    x1 = Conv2D(64, (3,3), padding='same', activation='relu', name='mod1_block1_conv2')(x1)
    x1_pool = MaxPooling2D((2,2), strides=(2,2), name='mod1_block1_pool')(x1)

    # ===================== 构建模型主体 =====================
    def build_block(prev_rgb, prev_flow, block_num, filters):
        # 修改2：正确传递concat参数
        merged = Lambda(lambda x: concation(x), name=f'merge_{block_num}')([prev_rgb, prev_flow])
        gap = GlobalAveragePooling2D(name=f'gap_{block_num}')(merged)
        
        # CCA损失
        cca_loss_layer = Lambda(
            lambda x: cca_loss(outdim_size, use_all_singular_values)(x[:, :tf.shape(x)[1]//2], x[:, tf.shape(x)[1]//2:]),
            name=f'cca_loss_block{block_num}'
        )(gap)
        
        # 注意力调整
        adj_rgb = Lambda(lambda x: cca_attention_1(x[0], x[1]),  # 显式拆分参数
                        name=f'att_rgb_{block_num}')([gap, prev_rgb])
        adj_flow = Lambda(lambda x: cca_attention_2(x[0], x[1]),
                        name=f'att_flow_{block_num}')([gap, prev_flow])
        
        return adj_rgb, adj_flow, cca_loss_layer

    # ===================== 逐块处理 =====================
    # Block1
    current_rgb, current_flow, loss1 = build_block(x0_pool, x1_pool, 1, 64)
    
    # Block2
    current_rgb = Conv2D(128, (3,3), padding='same', activation='relu', name='mod0_block2_conv1')(current_rgb)
    current_rgb = Conv2D(128, (3,3), padding='same', activation='relu', name='mod0_block2_conv2')(current_rgb)
    current_rgb = MaxPooling2D((2,2), name='mod0_block2_pool')(current_rgb)
    
    current_flow = Conv2D(128, (3,3), padding='same', activation='relu', name='mod1_block2_conv1')(current_flow)
    current_flow = Conv2D(128, (3,3), padding='same', activation='relu', name='mod1_block2_conv2')(current_flow)
    current_flow = MaxPooling2D((2,2), name='mod1_block2_pool')(current_flow)
    current_rgb, current_flow, loss2 = build_block(current_rgb, current_flow, 2, 128)

    # Block3
    current_rgb = Conv2D(256, (3,3), padding='same', activation='relu', name='mod0_block3_conv1')(current_rgb)
    current_rgb = Conv2D(256, (3,3), padding='same', activation='relu', name='mod0_block3_conv2')(current_rgb)
    current_rgb = Conv2D(256, (3,3), padding='same', activation='relu', name='mod0_block3_conv3')(current_rgb)
    current_rgb = MaxPooling2D((2,2), name='mod0_block3_pool')(current_rgb)
    
    current_flow = Conv2D(256, (3,3), padding='same', activation='relu', name='mod1_block3_conv1')(current_flow)
    current_flow = Conv2D(256, (3,3), padding='same', activation='relu', name='mod1_block3_conv2')(current_flow)
    current_flow = Conv2D(256, (3,3), padding='same', activation='relu', name='mod1_block3_conv3')(current_flow)
    current_flow = MaxPooling2D((2,2), name='mod1_block3_pool')(current_flow)
    current_rgb, current_flow, loss3 = build_block(current_rgb, current_flow, 3, 256)

    # Block4
    current_rgb = Conv2D(512, (3,3), padding='same', activation='relu', name='mod0_block4_conv1')(current_rgb)
    current_rgb = Conv2D(512, (3,3), padding='same', activation='relu', name='mod0_block4_conv2')(current_rgb)
    current_rgb = Conv2D(512, (3,3), padding='same', activation='relu', name='mod0_block4_conv3')(current_rgb)
    current_rgb = MaxPooling2D((2,2), name='mod0_block4_pool')(current_rgb)
    
    current_flow = Conv2D(512, (3,3), padding='same', activation='relu', name='mod1_block4_conv1')(current_flow)
    current_flow = Conv2D(512, (3,3), padding='same', activation='relu', name='mod1_block4_conv2')(current_flow)
    current_flow = Conv2D(512, (3,3), padding='same', activation='relu', name='mod1_block4_conv3')(current_flow)
    current_flow = MaxPooling2D((2,2), name='mod1_block4_pool')(current_flow)
    current_rgb, current_flow, loss4 = build_block(current_rgb, current_flow, 4, 512)

    # Block5
    current_rgb = Conv2D(512, (3,3), padding='same', activation='relu', name='mod0_block5_conv1')(current_rgb)
    current_rgb = Conv2D(512, (3,3), padding='same', activation='relu', name='mod0_block5_conv2')(current_rgb)
    current_rgb = Conv2D(512, (3,3), padding='same', activation='relu', name='mod0_block5_conv3')(current_rgb)
    current_rgb = MaxPooling2D((2,2), name='mod0_block5_pool')(current_rgb)
    
    current_flow = Conv2D(512, (3,3), padding='same', activation='relu', name='mod1_block5_conv1')(current_flow)
    current_flow = Conv2D(512, (3,3), padding='same', activation='relu', name='mod1_block5_conv2')(current_flow)
    current_flow = Conv2D(512, (3,3), padding='same', activation='relu', name='mod1_block5_conv3')(current_flow)
    current_flow = MaxPooling2D((2,2), name='mod1_block5_pool')(current_flow)
    _, _, loss5 = build_block(current_rgb, current_flow, 5, 512)

    # ===================== 分类头 =====================
    merged = Concatenate()([
        GlobalAveragePooling2D()(current_rgb),
        GlobalAveragePooling2D()(current_flow)
    ])
    x = Dense(4096, activation='relu', kernel_regularizer=l2(0.03))(merged)
    x = Dropout(0.3)(x)
    x = Dense(4096, activation='relu', kernel_regularizer=l2(0.03))(x)
    x = Dropout(0.2)(x)
    predictions = Dense(4, activation='softmax', name='classification')(x)

    # ===================== 模型编译 =====================
    model = Model(
        inputs=[img_input1, img_input2],
        outputs=[predictions, loss1, loss2, loss3, loss4, loss5]
    )

    # ===================== 加载VGG权重 =====================
    vgg_pretrained = VGG16(weights='imagenet', include_top=False)
    layer_mapping = {
        'mod0_block1_conv1': 'block1_conv1',
        'mod0_block1_conv2': 'block1_conv2',
        'mod0_block2_conv1': 'block2_conv1',
        'mod0_block2_conv2': 'block2_conv2',
        'mod0_block3_conv1': 'block3_conv1',
        'mod0_block3_conv2': 'block3_conv2',
        'mod0_block3_conv3': 'block3_conv3',
        'mod0_block4_conv1': 'block4_conv1',
        'mod0_block4_conv2': 'block4_conv2',
        'mod0_block4_conv3': 'block4_conv3',
        'mod0_block5_conv1': 'block5_conv1',
        'mod0_block5_conv2': 'block5_conv2',
        'mod0_block5_conv3': 'block5_conv3'
    }

    for vgg_layer in vgg_pretrained.layers:
        if vgg_layer.name in layer_mapping.values():
            target_name = [k for k,v in layer_mapping.items() if v == vgg_layer.name][0]
            try:
                model.get_layer(target_name).set_weights(vgg_layer.get_weights())
                print(f"成功加载权重: {target_name} <- {vgg_layer.name}")
            except (ValueError, IndexError) as e:
                print(f"权重加载失败 {target_name}: {str(e)}")

    # ===================== 损失配置 =====================
    loss_dict = {'classification': 'categorical_crossentropy'}
    loss_weights = {'classification': 1.0}
    for i in range(1,6):
        loss_dict[f'cca_loss_block{i}'] = lambda y_true, y_pred: y_pred
        loss_weights[f'cca_loss_block{i}'] = 0.001

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss=loss_dict,
        loss_weights=loss_weights,
        metrics={'classification': 'accuracy'}
    )

    return model