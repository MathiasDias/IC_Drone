#trocar os cód RGB por cod das classes
#class1=groundtruth, class2=panicum, class3=grass, class4=ground

from PIL import Image
#primeiro coloca a imagem original criada por nós
input_image = Image.open("/media/gabsp00/4202077D02077567/Backup_ssd/Gabslau/TCC/ambvirlx/img_cmp_old/n000000001")
input_image =  input_image.convert("RGB")
pixel_map = input_image.load()
width, height = input_image.size

lista_of=[]
lista_gn=[]

for i in range(width):
    for j in range(height):
        lista_of.append(pixel_map[i, j])

i=0
while i<len(lista_of):
	if lista_of[i] == (255,0,0):
		lista_of[i]=3
	elif lista_of[i] == (255,255,0):
		lista_of[i]=4
	elif lista_of[i] == (0,255,0):
		lista_of[i]=2
	elif lista_of[i] == (0,0,0):
		lista_of[i]=1
	i+=1

#agora é a imagem gerada pelo código
input_image = Image.open("/media/gabsp00/4202077D02077567/Backup_ssd/Gabslau/TCC/ambvirlx/img_cmp_old/n1")
input_image =  input_image.convert("RGB")
pixel_map2 = input_image.load()
width, height = input_image.size

for i in range(width):
    for j in range(height):
        lista_gn.append(pixel_map2[i, j])

i=0

while i<len(lista_gn):
	if lista_gn[i] == (255,0,0):
		lista_gn[i]=3
	elif lista_gn[i] == (255,255,0):
		lista_gn[i]=4
	elif lista_gn[i] == (0,255,0):
		lista_gn[i]=2
	elif lista_gn[i] == (0,0,0):
		lista_gn[i]=1
	i+=1

list_a=[]
list_b=[]
clas=1

while clas<=4:
	i,a,b=0,0,0
	while i<len(lista_of):
		if lista_of[i]==clas or lista_gn[i]==clas:
			a+=1
			if lista_of[i]==lista_gn[i]:
				b+=1
		i+=1
	list_a.append(a)
	list_b.append(b)
	clas+=1
	

miou_list=[]
iou=0
m=0

while m<4:
	print("Classe: ", m, " // ",list_a[m]," // ", list_b[m])
	miou_list.append(list_b[m]/list_a[m])
	iou=iou+miou_list[m]
	m+=1

miou_g=iou/4

print("Este é o teste da figura 1\nEste e o miou geral",miou_g)
print("Este e o miou do não classificado: ", miou_list[0], " // Este e o do panicum: ", miou_list[1], " // Este o da grama: ", miou_list[2]," // E o do chão: ", miou_list[3])
