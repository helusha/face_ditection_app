
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore # 環境設定をした環境下で読み込みできる

model_xml = "./model/face-detection-adas-0001.xml" # 学習モデルの呼び出し
model_bin = "./model/face-detection-adas-0001.bin" # xmlに対応するbinファイル

ie = IECore() # IRの作成
net = IENetwork(model=model_xml, weights=model_bin) # モデルの読み込み、定義
input_blob = next(iter(net.inputs)) # モデルの入力パラメータを取り出す
out_blob = next(iter(net.outputs)) # 出力パラメータを取り出す
net.batch_size = 1 # バッチサイズ（周りのやつ）
net.inputs[input_blob].precision = "U8" # uint8型(画像配列)に変換
n, c, h, w = net.inputs[input_blob].shape # モデルで必要とされる枚数n､深さc､幅高さ
#image = np.ndarray(shape=(c, h, w)) # 配列を準備

image = cv2.imread('face.png') # 画像の読み込み

if image.shape[:-1] != (w ,h): # 大きさが違うときリサイズ
    image = cv2.resize(image, (w, h))
image = image.transpose((2, 0, 1))  # 入力画像がHWC なのをCHWの変換

exec_net = ie.load_network(network=net, device_name='CPU') # 設定したプラグインの読み込み
res = exec_net.infer(inputs={input_blob: image}) # 推論を行う
res = res[out_blob] # 結果の取り出し

_, _, out_h, out_w = res.shape # 4次元の時の大きさ



print(image.shape)
for face in res[0][0]:
    if face[2]>0.5:
        box = face[3:7] * np.array([w, h, w, h])
        (xmin, ymin, xmax, ymax) = box.astype("int")
        #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        print(xmin, ymin, xmax, ymax)
        print("書き込みました")

print(image.dtype)
print(image.shape)

img = cv2.imread("face.png")
if img.shape[:-1] != (w ,h): # 大きさが違うときリサイズ
    img = cv2.resize(img, (w, h))

cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
cv2.imshow('image',img)
cv2.waitKey()
cv2.destroyAllWindows()
