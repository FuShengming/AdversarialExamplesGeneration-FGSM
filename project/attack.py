import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
import time

def getModel(path):
    model = keras.models.load_model(path)
    return model

def fgsmAttack(image,model):

    image = image
    image = np.expand_dims(image, axis=0)
    ##print(type(org_img))
    advImage = np.copy(image)
    inputLayer = model.layers[0].input
    outputLayer = model.layers[-1].output

    # 最大变化值
    maxDiff = 0.02
    above = image + maxDiff
    below = image - maxDiff
    ##np.reshape(org_img,(-1,28,28))

    prediction = model.predict(image)
    originPredictionLabel = np.argmax(prediction)

    # 获取损失函数，梯度函数和函数实例
    costFunc = outputLayer[0, originPredictionLabel]
    gradientFunc = tf.gradients(costFunc, inputLayer)[0]
    funcInstance = K.function([inputLayer, K.learning_phase()],
                          [costFunc, gradientFunc])

    #手动设置学习率
    learningRate = 0.0005
    count = 1
    change = 0.01

    while np.argmax(model.predict(advImage)) == originPredictionLabel:

        if count % 50 == 0:
            below -= change
            above += change
        cost, gradient = funcInstance([advImage, 0])
        n = np.sign(gradient)
        advImage -= n * learningRate
        advImage = np.clip(advImage, below, above)
        advImage = np.clip(advImage, 0, 1.0)
        count += 1
    advPredictionLabel = np.argmax(model.predict(advImage))
    res=0
    if(originPredictionLabel==advPredictionLabel):
        res=1

    advImage = advImage.squeeze(0)

    return [advImage,res]

def attackMain(images,shape):
    model=getModel("./model.h5")
    test_images=images
    length=shape[0]
    np.reshape(test_images,(length,28,28))
    p=model.predict(test_images)
    count=0
    startTime=time.time()
    indexStart=0
    indexEnd=length
    totalTime=0
    attack_data=[]
    for i in range(indexStart,indexEnd):
        print("目前生成第 "+str(i)+" 个对抗样本")
        res=fgsmAttack(test_images[i],model)
        if res[1]!=1:
            count+=1


        attack_data.append(res[0])
        totalTime = time.time() - startTime
        print("当前样本生成完毕，当前共消耗时间：" + str(totalTime) + "秒")
    temp=np.array(attack_data)
    np.reshape(temp,shape)
   ## np.save("./attack/attack_data.npy",temp)
    totalTime=time.time()-startTime
    print("总消耗时间："+str(totalTime)+"秒")
    return temp
