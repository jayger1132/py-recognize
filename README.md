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

