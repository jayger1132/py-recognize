# Python圖像辨識專案
[![Quality gate](https://sonarqube.210.mlc.app/api/project_badges/quality_gate?project=py-recognize-26)](https://sonarqube.210.mlc.app/dashboard?id=py-recognize-26)

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
##### 參考資料
##### https://www.runoob.com/python/python-func-open.html
##### https://www.huaweicloud.com/articles/5b5c98238d126a90ca6d963e06cc9c06.html
##### https://leemeng.tw/practical-pandas-tutorial-for-aspiring-data-scientists.html
