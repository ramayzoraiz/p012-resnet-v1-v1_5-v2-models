import tensorflow as tf
from  tensorflow import keras as k
l=k.layers
"""In case of resnt50+ due to comparison with tf; 
for v1 v1_5 versions(50 layers plus), bias is set to true and 
for v2(50 layers plus), bias=False in only first two conv2d(named conv1 and conv2) of bottleneck_block function.
In case of basic_block, bias is set to false for all conv2d layers."""

def basic_block(x, v, filters, kernel_size=3, stride=1, conv_shortcut=False, use_bias=False, name=None):
    if conv_shortcut:
        name = name+"_conv_shortcut/"
    else:
        name = name+"_maxpool_shortcut/" if stride>1 else name+"_id_shortcut/"
    
    if v==2.0: 
        preact = l.BatchNormalization(epsilon=1.001e-5, name=name + "preact_bn")(x)
        preact = l.Activation("relu", name=name + "preact_relu")(preact)
    
    if conv_shortcut:
        if v==2.0:
            shortcut = l.Conv2D(filters, 1, strides=stride, use_bias=use_bias, name=name + "conv0")(preact)
        else: # case v==1.0
            shortcut = l.Conv2D(filters, 1, strides=stride, use_bias=use_bias, name=name + "conv0")(x)
            shortcut = l.BatchNormalization(name=name + 'bn0')(shortcut)
    else:
        shortcut = (l.MaxPooling2D(1, strides=stride, name=name+'maxpool0')(x) if stride > 1 else x)

    if v==2.0: x=preact
    x = l.Conv2D(filters, kernel_size, strides=stride, name=name + "conv1", use_bias=use_bias, padding='same')(x)
    x = l.BatchNormalization(epsilon=1.001e-5, name=name + "bn1")(x)
    x = l.Activation("relu", name=name + "relu1")(x)

    x = l.Conv2D(filters, kernel_size, strides=1, padding='same', name=name + "conv2", use_bias=use_bias)(x)

    if v<2.0:
        x = l.BatchNormalization(epsilon=1.001e-5, name=name + "bn2")(x)
        x = l.Add(name=name + "add")([shortcut, x])
        x = l.Activation("relu", name=name + "relu_out")(x)
    else:
        x = x = l.Add(name=name + "add_out")([shortcut, x])

    return x

def bottleneck_block(x, v, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    if conv_shortcut:
        name = name+"_conv_shortcut/"
    else:
        name = name+"_maxpool_shortcut/" if stride>1 else name+"_id_shortcut/"
    
    if v==2.0: 
        preact = l.BatchNormalization(epsilon=1.001e-5, name=name + "preact_bn")(x)
        preact = l.Activation("relu", name=name + "preact_relu")(preact)
    
    if conv_shortcut:
        if v==2.0:
            shortcut = l.Conv2D(4 * filters, 1, strides=stride, name=name + "conv0")(preact)
        else: # case v==1.0 or 1.5
            shortcut = l.Conv2D(4 * filters, 1, strides=stride, name=name + "conv0")(x)
            shortcut = l.BatchNormalization(name=name + 'bn0')(shortcut)
    else:
        shortcut = (l.MaxPooling2D(1, strides=stride, name=name+'maxpool0')(x) if stride > 1 else x)

    if v==1.0:
        s1=stride; s2=1
    else: # case v==1.5 or 2.0
        s1=1; s2=stride

    if v==2.0:
        bias=False
    else:
        bias=True
    if v==2.0: x=preact
    x = l.Conv2D(filters, 1, strides=s1, name=name + "conv1", use_bias=bias)(x)
    x = l.BatchNormalization(epsilon=1.001e-5, name=name + "bn1")(x)
    x = l.Activation("relu", name=name + "relu1")(x)

    x = l.Conv2D(filters, kernel_size, strides=s2, padding='same', name=name + "conv2", use_bias=bias)(x)
    x = l.BatchNormalization(epsilon=1.001e-5, name=name + "bn2")(x)
    x = l.Activation("relu", name=name + "relu2")(x)

    x = l.Conv2D(4 * filters, 1, name=name + "conv3")(x)
    if v<2.0:
        x = l.BatchNormalization(epsilon=1.001e-5, name=name + "bn3")(x)
        x = l.Add(name=name + "add")([shortcut, x])
        x = l.Activation("relu", name=name + "relu_out")(x)
    else:
        x = x = l.Add(name=name + "add_out")([shortcut, x])

    return x


def stack_blocks(x, v, filters, num_blocks, block, stride=2, name=None, first_conv=True):
    """A set of stacked residual blocks.

    Args:
        x: Input tensor.
        filters: Number of filters in the bottleneck layer in a block.
        blocks: Number of blocks in the stacked blocks.
        stride1: Stride of the first layer in the first block. Defaults to `2`.
        name: Stack label.

    Returns:
        Output tensor for the stacked blocks.
    """

    if v<2:
        s1=stride; s3=1
    else:
        s1=1; s3=stride
        
    x = block(x, v, filters, stride=s1, conv_shortcut=first_conv, name=name+"b1")
    for i in range(2, num_blocks):
        x = block(x, v, filters, name=name + "b" + str(i))
    x = block(x, v, filters, stride=s3, conv_shortcut=False, name=name + "b" + str(num_blocks) )
    return x


def ResNet( stack_fn, v, include_top=True, input_shape=(224, 224, 3), num_classes=1000, name="resnet",):
    """Instantiates the ResNet, ResNetv1_5 and ResNetV2 architecture.
    Args:
        stack_fn: A function that returns output tensor for the
            stacked residual blocks.
        v: version number, only 1.0, 1.5, 2 allowed
        include_top: Whether to include the fully-connected
            layer at the top of the network else output is before averagepool.
        input_shape: tuple of 3 ints. Default(224, 224, 3)
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`.
        name: The name of the model (string).
    Returns:
        A Model instance.
    """
    
    img_input = l.Input(shape=input_shape)
    x = l.ZeroPadding2D(padding=3, name="conv1_pad")(img_input)
    x = l.Conv2D(64, 7, strides=2, name="conv1_conv")(x)
    if v<2.0: # case v==1.0 or v==1.5
        x = l.BatchNormalization(epsilon=1.001e-5, name="conv1_bn")(x)
        x = l.Activation("relu", name="conv1_relu")(x)
    x = l.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x)
    x = l.MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    x = stack_fn(x)

    if v==2.0:
        x = l.BatchNormalization(epsilon=1.001e-5, name="post_bn")(x)
        x = l.Activation("relu", name="post_relu")(x)
    if include_top:
        x = l.GlobalAveragePooling2D(name="avg_pool")(x)
        x = l.Dense(num_classes, activation='softmax', name="predictions")(x)

    # Create model.
    model = k.models.Model(inputs=img_input, outputs=x, name=name)
    return model


def ResNet50(v,
    include_top=True,
    input_shape=(224, 224, 3),
    num_classes=1000,
    name="resnet50",
):
    """Instantiates the ResNet50 architecture."""

    assert v in [1.0, 1.5, 2.0], f"(ResNet50 function) v={v} is not in the allowed list [1.0, 1.5, 2.0]"

    if v<2.0:
        s1=1; s4=2
    else:
        s1=2; s4=1
        
    def stack_fn(x, block=bottleneck_block):
        x = stack_blocks(x, v,  64, 3, block, stride=s1, name="g1_")
        x = stack_blocks(x, v, 128, 4, block, name="g2_")
        x = stack_blocks(x, v, 256, 6, block, name="g3_")
        x = stack_blocks(x, v, 512, 3, block, stride=s4, name="g4_")
        return x

    return ResNet(stack_fn, v, include_top=include_top, input_shape=input_shape,
        num_classes=num_classes, name=name)

def ResNet34(v,
    include_top=True,
    input_shape=(224, 224, 3),
    num_classes=1000,
    name="resnet34",
):
    """Instantiates the ResNet50 architecture."""

    assert v in [1.0, 2.0], f"(ResNet50 function) v={v} is not in the allowed list [1.0, 2.0]"

    if v<2.0:
        s1=1; s4=2; con=False
    else:
        s1=2; s4=1; con=True

        
    def stack_fn(x, block=basic_block):
        x = stack_blocks(x, v,  64, 3, block, stride=s1, first_conv=False, name="g1_")
        x = stack_blocks(x, v, 128, 4, block, name="g2_")
        x = stack_blocks(x, v, 256, 6, block, name="g3_")
        x = stack_blocks(x, v, 512, 3, block, stride=s4, name="g4_")
        return x

    return ResNet(stack_fn, v, include_top=include_top, input_shape=input_shape,
        num_classes=num_classes, name=name)

def ResNet18(v,
    include_top=True,
    input_shape=(224, 224, 3),
    num_classes=1000,
    name="resnet18",
):
    """Instantiates the ResNet50 architecture."""

    assert v in [1.0], f"(ResNet50 function) v={v} is not in the allowed list [1.0]"
        
    def stack_fn(x, block=basic_block):
        x = stack_blocks(x, v,  64, 2, block, stride=1, first_conv=False, name="g1_")
        x = stack_blocks(x, v, 128, 2, block, name="g2_")
        x = stack_blocks(x, v, 256, 2, block, name="g3_")
        x = stack_blocks(x, v, 512, 2, block, name="g4_")
        return x

    return ResNet(stack_fn, v, include_top=include_top, input_shape=input_shape,
        num_classes=num_classes, name=name)


