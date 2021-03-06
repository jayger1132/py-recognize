# Python圖像辨識專案
[![Quality gate](https://sonarqube.210.mlc.app/api/project_badges/quality_gate?project=py-recognize-26)](https://sonarqube.210.mlc.app/dashboard?id=py-recognize-26)
# 錄製影片時要注意攝影機是否有定位，我錄影的方式是確定攝影機的絕對位置，讓鏡頭下緣對齊地上規範的線

## 設定dir必須要以英文為主不然抓path很容易出問題
#### 輸出放入的檔案 cv2.imshow('',img)
```py
cam = cv2.VideoCapture(video_path)
image = cam.read()
cv2.imshow('test',image)
```
#### 要判斷dir是否存在可以用os.exists(path)
```py
if not os.path.exists(save_path):
    os.mkdir(save_path)
```
#### 新增file的話可以使用os.open就可以了/csv.writerow寫一行
```py
# 'a+' 是附加在後面 'a'是在前面
with open('./data/csv/Biceps_curl.csv', 'a+', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)
    # 寫入另外幾列資料
    for i in range(0,24):
        writer.writerow([str(datum.poseKeypoints[0][i][0]),str(datum.poseKeypoints[0][i][1]),str(datum.poseKeypoints[0][i][2])])
    writer.writerow(["","",""])
```
#### 陣列extend V.S. append
```py
tmp_data=[]
# range(0,25) => 0~24
for i in range(0,25):
    #extend 是[0],[1],[2]這樣+ append是[012],[012],[012]的+
    tmp_data.extend(['x','y','score'])
#print(tmp_data)
```
#### parser
```py
# Flags
parser = argparse.ArgumentParser()
# "-i"也可寫成簡寫
parser.add_argument("--image_dir", default="./data/imgs/from_video/Biceps_curl/img_0-99/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
#parse_known_args()使用時機是argument不只有一個，當命令中傳入之後才會用到的選項時不會報錯而是先存起來保留到之後使用 參考資料有
args = parser.parse_known_args()
#print(args[0].image_dir)
```
#### 辨識 單張/多張圖片 
```py
# Process 1
datum = op.Datum()
imageToProcess = cv2.imread(args[0].image_path)
datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop(op.VectorDatum([datum]))
# Process mutiple and display images 
imagePaths = op.get_images_on_directory(args[0].image_dir)
for imagePath in imagePaths:
    datum = op.Datum()
    imageToProcess = cv2.imread(imagePath)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    print(str(datum.poseKeypoints))
```
#### openpose辨識有可能會出現None的問題
```py
imageToProcess = cv2.imread(imagePath)
datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop(op.VectorDatum([datum]))
#print(str(datum.poseKeypoints))
#要先判斷圖片是否判斷得出來 會有None的問題
if datum.poseKeypoints is None:
    print(str(imagePath),"Noneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    try:
        os.remove(imagePath)
        continue
    except OSError as e:
        print(e)
else :
    print(imagePath)
```
#### DF用法 , loc/iloc 差在iloc通常都是用數字(index去取資料) loc是用"name"
```py
X=pd.DataFrame(data,columns=data.columns[:-1])
print (X.iloc[:,0:2]) #這樣是 columns0~2
print (X.iloc[:,::-1]) #這樣是 columns反過來輸出
print (X.iloc[::-1,:]) #這樣是 index反過來輸出
print (X.loc[:,"X"])
#單純輸出X沒辦法看到全部的資料，因為是concat合併的，就已經依照輸入的data來排列順序了
print(X)
```
#### 讀取的格式跑掉
```py
#有時候格式跑掉可以先去第一排+個字在重新run 我是在Score旁+一行，再回來執行
df_bending_test = pd.read_csv('./data/csv/Biceps_curl_Bending_Test.csv')
```
#### 合併資料
```py
#axis=0通常是直的合併 1為橫的
d = pd.concat(datas,axis=0)
```
#### train_test_split 前面有先把csv合併儲存成dataframe的格式
```py
#分出train,test。test_size=0.2 -> training : test = 8 : 2。random_state=None和random_state=0是不一樣的，若為None時，劃分出來的測試集或訓練集中，其類標籤的比例也是隨機的。random_state 隨機數的種子，種子相同就算例子不同也會產生相同的隨機數
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
```
#### 寫enviromet的方式可以參考
```py
##enviroment
class MyArgs():
    def __init__(self):
        self.video_path = './data/video/3.mp4'
        self.model = './data/model/model_1'
args = MyArgs()
```
#### 使用訓練模型
```py
#處理 Calling Model.predict in graph mode is not supported when the Model instance was constructed with eager mode enabled

#https://www.codeleading.com/article/42675321680/ 
#像是重新設定一次所以後面需要再重新load
model = keras.models.Sequential()
model.call = tf.function(model.call)
#08/24 發現根本不需要上述這兩行 一開始出錯只是因為沒有load進去
model = keras.models.load_model(args.model)
model.predict(np.ones((1, 75)))
#model.predict_classes(test)預測輸出的是類別 ，model.predict(test) 預測輸出的是數值
res = model.predict_classes(data)
```
#### 躺著的動作需要旋轉讓他是立著的狀態openpose比較好判斷
```py
def Rotate(img):
    
    h = img.shape[0]
    w = img.shape[1]
    center = (h/2,w/2)
    #rotate 旋轉中心座標 , "90"為逆時針90度, "-90"為順時針90度, 1為縮放1倍
    R = cv2.getRotationMatrix2D(center,-90,1)
    rotate = cv2.warpAffine(img,R,(w,h))
    return rotate
```
#### 用歐基里德距離分析影像相似度
```py
# 先分析目前屬於哪個階段->判斷在該階段上還是下->在該階段下面時與起始點(改階段-1)做比較;反之

#Score0 表示與0的距離
Score0 = math.pow(math.pow(np.linalg.norm(Ax[1] - tmp_AVGx),2)+math.pow(np.linalg.norm(Ay[1] - tmp_AVGy),2),0.5)
Score1 = math.pow(math.pow(np.linalg.norm(Ax[2] - tmp_AVGx),2)+math.pow(np.linalg.norm(Ay[2] - tmp_AVGy),2),0.5)
Score2 = math.pow(math.pow(np.linalg.norm(Ax[3] - tmp_AVGx),2)+math.pow(np.linalg.norm(Ay[3] - tmp_AVGy),2),0.5)
Score3 = math.pow(math.pow(np.linalg.norm(Ax[4] - tmp_AVGx),2)+math.pow(np.linalg.norm(Ay[4] - tmp_AVGy),2),0.5)
Score4 = math.pow(math.pow(np.linalg.norm(Ax[5] - tmp_AVGx),2)+math.pow(np.linalg.norm(Ay[5] - tmp_AVGy),2),0.5)
print("與等級0比較",Score0,"\n與等級1比較",Score1,"\n與等級2比較",Score2,"\n與等級3比較",Score3,"\n與等級4比較",Score4)
```
#### 二頭彎舉階段示意圖
![image](https://github.com/jayger1132/py-recognize/blob/main/img/%E4%BA%8C%E9%A0%AD%E5%BD%8E%E8%88%89%E9%9A%8E%E6%AE%B5%E5%9C%96.jpg)
#### 運動過程解析圖
![image](https://github.com/jayger1132/py-recognize/blob/main/img/%E9%81%8B%E5%8B%95%E9%81%8E%E7%A8%8B%E5%88%86%E9%A1%9E%E5%9C%96.jpg)
#### cv2作圖、編譯中文文字
```py
#Cv2輸出中文
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判斷是否OpenCV圖片類型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
if(Flag_action == "Ready" ):
    cv2.rectangle(target_out, (350, 10), (470, 60), (255, 255, 255), -1)
    cv2.putText(target_out, 'Pause', (360, 45), 4, 1, (255, 0, 0), 1, cv2.LINE_AA)
    
else :
    
    cv2.rectangle(target_out, (350, 10), (470, 90), (255, 255, 255), -1)
    target_out=cv2ImgAddText(target_out, "階段 "+str(LV_out), 360, 15, (255, 0, 0), 35)
    target_out=cv2ImgAddText(target_out, "分數 "+str(Text1), 360, 50, (255, 0, 0), 35)
#target_out=(cv2.cvtColor(target_out, cv2.COLOR_BGR2RGB))
cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", target_out)
cv2.waitKey(100)
```
#### 分析訓練模組 
```py
# 5_DNN_Test
#frac=0.2 隨機取總資料*0.2 當frac>1要設定是否要replace =>取出後不放回
def handleData(datas, frac ):
    od = pd.concat(datas, axis=0)
    
    if frac is not None:
        #frac 如果不設定成float numpy會出錯 格式不對
        d = od.sample(frac = float(frac))
    else:
        d = od
    ds = pd.DataFrame(d, columns=d.columns[:-1])
    tx = ds
    ty = to_categorical(d['target'], TARGET_DIM)
    return (tx, ty)
def testAcc(_model, frac = None):
    tx, ty = handleData([df_start, df_ready], frac)
    # batch_size = 1 一張一張比對
    score = _model.evaluate(tx, ty , batch_size=1)
    print ('Acc:', score[1])
    return score[1]
```
##### 參考資料
##### https://www.runoob.com/python/python-func-open.html
##### https://www.huaweicloud.com/articles/5b5c98238d126a90ca6d963e06cc9c06.html
##### https://leemeng.tw/practical-pandas-tutorial-for-aspiring-data-scientists.html
#### 模型使用
##### https://kknews.cc/zh-tw/code/6lvzmaq.html
##### https://ithelp.ithome.com.tw/articles/10197257
##### https://keras-cn.readthedocs.io/en/latest/models/model/
### 串流參考,建議
##### https://pacific-emmental-9bd.notion.site/29755d2e7eb84077a4b20d00e202dab7