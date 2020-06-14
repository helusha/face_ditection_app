
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore # 環境設定をした環境下で読み込みできる

model_xml = "./model/face-detection-adas-0001.xml" # 学習モデルの呼び出し
model_bin = "./model/face-detection-adas-0001.bin" # xmlに対応するbinファイル
model_age_xml = "./model/age-gender-recognition-retail-0013.xml" # 学習モデルの呼び出し
model_age_bin = "./model/age-gender-recognition-retail-0013.bin" # xmlに対応するbinファイル



#age版
ie_age = IECore() # IRの作成
net_age = IENetwork(model=model_age_xml, weights=model_age_bin) # モデルの読み込み、定義
input_blob_age = next(iter(net_age.inputs)) # モデルの入力パラメータを取り出す
out_blob_age = next(iter(net_age.outputs)) # 出力パラメータを取り出す
net_age.batch_size = 1 # バッチサイズ（周りのやつ）
net_age.inputs[input_blob_age].precision = "U8" # uint8型(画像配列)に変換
n, c, h_age, w_age = net_age.inputs[input_blob_age].shape # モデルで必要とされる枚数n､深さc､幅高さ
#image = np.ndarray(shape=(c, h, w)) # 配列を準備

image = cv2.imread('me.png') # 画像の読み込み
if image.shape[:-1] != (w_age ,h_age): # 大きさが違うときリサイズ
    image = cv2.resize(image, (w_age, h_age))
image = image.transpose((2, 0, 1))  # 入力画像がHWC なのをCHWの変換

exec_net_age = ie_age.load_network(network=net_age, device_name='CPU') # 設定したプラグインの読み込み
res_age = exec_net_age.infer(inputs={input_blob_age: image}) # 推論を行う
res_age = res_age[out_blob_age] # 結果の取り出し
prob = exec_net_age.requests[0].outputs['prob']
#print("prob",prob)
label = ('Female', 'Male') #男女のラベル
gender = label[np.argmax(prob[0])]
#print(gender)
#print("年齢",res_age*100)
age = int(res_age[0][0][0][0]*100)


#顔の輪郭
ie = IECore() # IRの作成
net = IENetwork(model=model_xml, weights=model_bin) # モデルの読み込み、定義
input_blob = next(iter(net.inputs)) # モデルの入力パラメータを取り出す
out_blob = next(iter(net.outputs)) # 出力パラメータを取り出す
net.batch_size = 1 # バッチサイズ（周りのやつ）
net.inputs[input_blob].precision = "U8" # uint8型(画像配列)に変換
n, c, h, w = net.inputs[input_blob].shape # モデルで必要とされる枚数n､深さc､幅高さ
#image = np.ndarray(shape=(c, h, w)) # 配列を準備
image2 = cv2.imread('me.png') # 画像の読み込み
if image2.shape[:-1] != (w ,h): # 大きさが違うときリサイズ
    image2 = cv2.resize(image2, (w, h))
image2 = image2.transpose((2, 0, 1))  # 入力画像がHWC なのをCHWの変換

exec_net = ie.load_network(network=net, device_name='CPU') # 設定したプラグインの読み込み
res = exec_net.infer(inputs={input_blob: image2}) # 推論を行う
res = res[out_blob] # 結果の取り出し

_, _, out_h, out_w = res.shape # 4次元の時の大きさ

print(res)


for face in res[0][0]:
    if face[2]>0.5:
        box = face[3:7] * np.array([w, h, w, h])
        (xmin, ymin, xmax, ymax) = box.astype("int")
        #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        print(xmin, ymin, xmax, ymax)
        print("書き込みました")


img = cv2.imread("me.png")
if img.shape[:-1] != (672, 384): # 大きさが違うときリサイズ
    img = cv2.resize(img, ( 672, 384))
    img = cv2.putText(img, "age:"+str(age)+ " "+ str(gender) ,(xmin, ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),1)
cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

cv2.imshow('image',img)
cv2.waitKey()
cv2.destroyAllWindows()
