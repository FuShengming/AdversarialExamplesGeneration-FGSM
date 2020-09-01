import attack
def generate(images,shape):
	if not(images.min()==0 and images.max==1):
		print("数据像素点取值范围不是[0,1],已将数据进行标准化")
		images.astype(float)
		images = images / float(images.max())
	return attack.attackMain(images,shape)