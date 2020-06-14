import cv2
image = cv2.imread('face.jpg') # 画像の読み込み
if image.shape[:-1] != (672, 384): # 大きさが違うときリサイズ
    image = cv2.resize(image, ( 672, 384))

cv2.rectangle(image, (295, 57), (395, 199), (0, 255, 0), 2)
cv2.imshow('image',image)
cv2.waitKey()
cv2.destroyAllWindows()