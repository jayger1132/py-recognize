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
with open('./data/csv/Biceps_curl.csv', 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        # 寫入一列資料
        writer.writerow(['X', 'Y', 'Score'])
        # 寫入另外幾列資料
        for i in range(0,24):
            writer.writerow([str(datum.poseKeypoints[0][i][0]),str(datum.poseKeypoints[0][i][1]),str(datum.poseKeypoints[0][i][2])])
```
